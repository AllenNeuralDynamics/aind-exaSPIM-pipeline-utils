"""Wrapper functions and runtime arguments definition."""
import datetime
import json
import logging
import os
import re
import selectors
import shutil
import subprocess
import sys
from typing import Dict, List, Union
from urllib.parse import urlparse

import argschema
import argschema.fields as fld
import marshmallow as mm
import psutil
import s3fs
from aind_data_schema.processing import DataProcess, ProcessName
import xml.etree.ElementTree as ET

# from aind_data_schema import DataProcess
# from aind_data_schema.processing import ProcessName


from . import __version__
from .imagej_macros import ImagejMacros
from .exaspim_manifest import get_capsule_manifest, write_process_metadata, ExaspimProcessingPipeline
from .qc import bigstitcher_log_edge_analysis
from .qc.create_ng_link import create_ng_link


class IPDetectionSchema(argschema.ArgSchema):  # pragma: no cover
    """Adjustable parameters to detect IP."""

    downsample = fld.Int(
        required=True,
        metadata={"description": "Downsampling factor. Use the one that is available in the dataset."},
    )
    bead_choice = fld.String(
        required=True,
        validate=mm.validate.OneOf(list(ImagejMacros.MAP_BEAD_CHOICE.keys())),
        metadata={"description": "Beads detection mode"},
    )
    sigma = fld.Float(
        load_default=1.8, metadata={"description": "Difference of Gaussians sigma (beads_mode==manual only)."}
    )
    threshold = fld.Float(
        load_default=0.1,
        metadata={"description": "Difference of Gaussians detection threshold (beads_mode==manual only)."},
    )
    find_minima = fld.Boolean(
        load_default=False, metadata={"description": "Find minima (beads_mode==manual only)."}
    )
    find_maxima = fld.Boolean(
        load_default=True, metadata={"description": "Find maxima (beads_mode==manual only)."}
    )
    set_minimum_maximum = fld.Boolean(
        load_default=False,
        metadata={"description": "Define the minimum and maximum intensity range manually"},
    )
    minimal_intensity = fld.Float(
        load_default=0, metadata={"description": "Minimal intensity value (if set_minimum_maximum==True)."}
    )
    maximal_intensity = fld.Float(
        load_default=65535,
        metadata={"description": "Minimal intensity value (if set_minimum_maximum==True)."},
    )
    maximum_number_of_detections = fld.Int(
        load_default=0,
        metadata={
            "description": "If not equal to 0, the number of maximum IPs to detect."
                           " Set ip_limitation_choice, too."
        },
    )
    ip_limitation_choice = fld.String(
        required=True,
        validate=mm.validate.OneOf(list(ImagejMacros.MAP_IP_LIMITATION_CHOICE.keys())),
        metadata={
            "description": "How to pick limit_amount_of_detections is set >0 and the maximum number is hit."
        },
    )


class IPRegistrationSchema(argschema.ArgSchema):  # pragma: no cover
    """Adjustable parameters to register with translation only."""

    transformation_choice = fld.String(
        required=True,
        validate=mm.validate.OneOf(list(ImagejMacros.MAP_TRANSFORMATION.keys())),
        metadata={"description": "Translation, rigid or full affine transformation ?"},
    )

    compare_views_choice = fld.String(
        required=True,
        validate=mm.validate.OneOf(list(ImagejMacros.MAP_COMPARE_VIEWS.keys())),
        metadata={"description": "Which views to compare ?"},
    )

    interest_point_inclusion_choice = fld.String(
        required=True,
        validate=mm.validate.OneOf(list(ImagejMacros.MAP_INTEREST_POINT_INCLUSION.keys())),
        metadata={"description": "Which interest points to use ?"},
    )

    fix_views_choice = fld.String(
        required=True,
        validate=mm.validate.OneOf(list(ImagejMacros.MAP_FIX_VIEWS.keys())),
        metadata={"description": "Which views to fix ?"},
    )

    fixed_tile_ids = fld.List(
        fld.Int,
        load_default=[
            0,
        ],
        metadata={"description": "Setup ids of fixed tiles (fix_views_choice==select_fixed)."},
    )
    map_back_views_choice = fld.String(
        required=True,
        validate=mm.validate.OneOf(list(ImagejMacros.MAP_MAP_BACK_VIEWS.keys())),
        metadata={"description": "How to map back views?"},
    )
    map_back_reference_view = fld.Int(
        load_default=0, metadata={"description": "Selected reference view for map back."}
    )
    do_regularize = fld.Boolean(default=False, metadata={"description": "Do regularize transformation?"})
    regularization_lambda = fld.Float(
        load_default=0.1, metadata={"description": "Regularization lambda (do_regularize==True only)."}
    )
    regularize_with_choice = fld.String(
        load_default="rigid",
        validate=mm.validate.OneOf(list(ImagejMacros.MAP_REGULARIZATION.keys())),
        metadata={"description": "Which regularization to use (do_regularize==True only) ?"},
    )


class ImageJWrapperSchema(argschema.ArgSchema):  # pragma: no cover
    """Command line arguments."""

    session_id = fld.String(required=True, metadata={"description": "Processing run session identifier"})
    memgb = fld.Int(
        required=True,
        metadata={
            "description": "Allowed Java interpreter memory. "
                           "Should be about 0.8 GB x number of parallel threads less than total available."
        },
    )
    parallel = fld.Int(
        required=True,
        metadata={"description": "Number of parallel Java worker threads."},
        validate=mm.validate.Range(min=1, max=128),
    )
    dataset_xml = fld.String(required=True, metadata={"description": "Input xml dataset definition"})
    do_detection = fld.Boolean(required=True, metadata={"description": "Do interest point detection?"})
    ip_detection_params = fld.Nested(
        IPDetectionSchema, required=False, metadata={"description": "Interest point detection parameters"}
    )
    do_registrations = fld.Boolean(
        required=True,
        metadata={"description": "Do first transformation fitting ?"},
    )
    ip_registrations_params = fld.Nested(
        IPRegistrationSchema,
        required=False,
        metadata={"description": "Registration parameters (do_registrations==True only)"},
        many=True,
    )


def print_output(data, f, stderr=False) -> None:  # pragma: no cover
    """Print output to stdout or stderr and to a file if specified."""
    if stderr:
        print(data, end="", file=sys.stderr, flush=True)
    else:
        print(data, end="", flush=True)
    if f:
        print(data, end="", file=f, flush=True)


def fmt_uri(uri: str, trailing_slash=True) -> str:  # pragma: no cover
    """Format the uri to be used as a folder name.

    All multiple occurrence internal slashes are replaced to single ones.

    Local paths can be relative or absolute, trailing slash will be added if missing.

    S3 references formatted to pattern ``s3://bucket/path/``
    """
    p = urlparse(uri)
    s1 = f"{p.scheme}:" if p.scheme else ""
    s2 = f"//{p.netloc}" if p.netloc else ""
    tslash = "/" if trailing_slash else ""
    s3 = re.sub(r"/{2,}", r"/", p.path.rstrip("/"))
    return s1 + s2 + s3 + tslash


def wrapper_cmd_run(cmd: Union[str, List], logger: logging.Logger, f_stdout=None, f_stderr=None) -> int:
    """Wrapper for a shell command.

    Wraps a shell command.

    It monitors, captures and re-prints stdout and strderr as the command progresses.

    TBD: Validate the program output on-the-fly and kill it if failure detected.

    Parameters
    ----------
    cmd: `str`
        Command that we want to execute.

    logger: `logging.Logger`
        Logger instance to use.

    Returns
    -------
    r: `int`
      Cmd return code.
    """
    logger.info("Starting command (%s)", str(cmd))
    with selectors.DefaultSelector() as sel, subprocess.Popen(
            cmd, bufsize=4096, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as p:
        sel.register(p.stdout, selectors.EVENT_READ)
        sel.register(p.stderr, selectors.EVENT_READ)
        while p.poll() is None:  # pragma: no cover
            for key, _ in sel.select():
                data = key.fileobj.read1().decode()
                if not data:
                    continue
                if key.fileobj is p.stdout:
                    print_output(data, f_stdout, stderr=False)
                else:
                    print_output(data, f_stderr, stderr=True)
        # Ensure to process everything that may be left in the buffer
        # This may be unnecessary, the process may not terminate until the pipes are at EOF
        data = p.stdout.read().decode()
        if data:
            print_output(data, f_stdout, stderr=False)
        data = p.stderr.read().decode()
        if data:
            print_output(data, f_stderr, stderr=True)
        r = p.poll()
    logger.info("Command finished with return code %d", r)
    return r


def get_auto_parameters(args: Dict) -> Dict:  # pragma: no cover
    """Determine environment parameters.

    Determine number of cpus, imagej memory limit and imagej macro file names.

    Note
    ----

    Number of cpus and memory are for the whole VM at the moment, not what is available
    for the capsule.

    Parameters
    ----------
    args: `Dict`
        ArgSchema args dictionary

    Returns
    -------
    params: `Dict`
      New dictionary with determined parameters.

    """
    ncpu = os.cpu_count()

    mem_GB = psutil.virtual_memory().total // (1024 * 1024 * 1024)
    d = int(mem_GB * 0.1)
    if d < 5:
        d = 5
    mem_GB -= d
    if mem_GB < 10:
        raise ValueError("Too little memory available")

    process_xml = "../results/bigstitcher.xml"
    macro_ip_det = "../results/macro_ip_det.ijm"
    return {
        "process_xml": process_xml,
        # Do not use, this is the whole VM at the moment, not what is available for the capsule
        "auto_ncpu": ncpu,
        "auto_memgb": mem_GB,
        "macro_ip_det": macro_ip_det,
    }


def main():  # pragma: no cover
    """Entry point if run as a standalone program. This uses the old-style config."""
    logging.basicConfig(format="%(asctime)s %(levelname)-7s %(message)s")

    logger = logging.getLogger()
    parser = argschema.ArgSchemaParser(schema_type=ImageJWrapperSchema)

    args = dict(parser.args)
    logger.setLevel(args["log_level"])
    args.update(get_auto_parameters(args))
    logger.info("Invocation: %s", sys.argv)

    logger.info("Writing out config.json")
    with open("/results/config.json", "w") as f:
        json.dump(args, f, indent=2)

    logger.info("Copying input xml %s -> %s", args["dataset_xml"], args["process_xml"])
    shutil.copy(args["dataset_xml"], args["process_xml"])

    if args["do_detection"]:
        det_params = dict(args["ip_detection_params"])
        det_params["parallel"] = args["parallel"]
        det_params["process_xml"] = args["process_xml"]
        logger.info("Creating macro %s", args["macro_ip_det"])
        with open(args["macro_ip_det"], "w") as f:
            f.write(ImagejMacros.get_macro_ip_det(det_params))
        r = wrapper_cmd_run(
            [
                "ImageJ",
                "-Dimagej.updater.disableAutocheck=true",
                "--headless",
                "--memory",
                "{memgb}G".format(**args),
                "--console",
                "--run",
                args["macro_ip_det"],
            ],
            logger,
        )
        if r != 0:
            raise RuntimeError("IP detection command failed.")
    else:
        if args["do_registrations"]:
            # We assume that interest point detections are already present in the input dataset
            # in the folder of the xml dataset file
            logger.info("Assume already detected interestpoints.")
            ip_src = os.path.join(os.path.dirname(args["dataset_xml"]), "interestpoints.n5")
            logger.info("Copying %s -> /results/", ip_src)
            shutil.copytree(ip_src, "/results/interestpoints.n5", dirs_exist_ok=True)

    if args["do_registrations"]:
        if "ip_registrations_params" not in args:
            raise ValueError("Registration steps are requested but no configuration provided.")
        reg_index = 0
        for reg_params in args["ip_registrations_params"]:
            macro_reg = f"/results/macro_ip_reg{reg_index:d}.ijm"
            reg_params = dict(reg_params)
            reg_params["process_xml"] = args["process_xml"]
            reg_params["parallel"] = args["parallel"]
            logger.info("Creating macro %s", macro_reg)
            with open(macro_reg, "w") as f:
                f.write(ImagejMacros.get_macro_ip_reg(reg_params))
            r = wrapper_cmd_run(
                [
                    "ImageJ",
                    "-Dimagej.updater.disableAutocheck=true",
                    "--headless",
                    "--memory",
                    "{memgb}G".format(**args),
                    "--console",
                    "--run",
                    macro_reg,
                ],
                logger,
            )
            if r != 0:
                raise RuntimeError("IP registration1 command failed.")
            reg_index += 1

    logger.info("Done.")


def get_imagej_wrapper_metadata(
        parameters: dict, input_location: str = None, output_location: str = None
) -> DataProcess:  # pragma: no cover
    """Initiate metadata instance with current timestamp and configuration."""
    t = datetime.datetime.now()
    dp = DataProcess(
        name=ProcessName.IMAGE_TILE_ALIGNMENT,
        software_version="0.1.0",
        start_date_time=t,
        end_date_time=t,
        input_location=input_location,
        output_location=output_location,
        code_url="https://github.com/AllenNeuralDynamics/aind-exaSPIM-pipeline-utils",
        code_version=__version__,
        parameters=parameters,
        outputs=None,
        notes="IN PROGRESS",
    )
    return dp


def set_metadata_done(meta: DataProcess) -> None:  # pragma: no cover
    """Update end timestamp and set metadata note to ``DONE``.

    Parameters
    ----------
    meta: DataProcess
      Capsule metadata instance.
    """
    t = datetime.datetime.now()
    meta.end_date_time = t
    meta.notes = "DONE"


def upload_alignment_results(args: dict):  # pragma: no cover
    """Upload the whole contents of the result folder to S3."""
    # Set up the S3 file system
    fs = s3fs.S3FileSystem()
    url = urlparse(args["output_uri"])
    if url.scheme != "s3":
        raise NotImplementedError("Only s3 output_uri is supported, not {url.scheme}")
    fs.put(
        "../results/", url.netloc + url.path.rstrip("/") + "/", recursive=True, maxdepth=10
    )  # Interestpoints.n5 have a bunch of subfolders


def create_emr_ready_xml(args: dict, num_regs: int = 1):  # pragma: no cover
    """Copy the solution xml into an EMR run ready version.

    We process one more xml than the number of registrations to have an emr xml
    for the original placements, too.
    """
    # read an xml file search for the zarr entry and replace it
    # supposedly handles utf-8 by default
    for i in range(num_regs + 1):
        suffix = f"~{i}" if i > 0 else ""
        tree = ET.parse(args["process_xml"] + suffix)
        emr_xml_name = "bigstitcher_emr_{}_{}_{}.xml".format(args["subject_id"], args["session_id"], i)
        root = tree.getroot()
        imgloader = root.find("SequenceDescription/ImageLoader")
        url = urlparse(args["input_uri"])
        s3b = ET.Element("s3bucket")
        s3b.text = url.netloc
        imgloader.insert(0, s3b)
        elem_zarr = imgloader.find("zarr")
        # substitute regex pattern in the beginning of elem_zarr.text
        elem_zarr.text = "/" + url.path.strip("/") + "/SPIM.ome.zarr"
        # write the xml file
        tree.write(f"../results/{emr_xml_name}", encoding="utf-8")


def create_edge_connectivity_report(num_registrations: int) -> None:  # pragma: no cover
    """Create a report of edge connectivity failures."""
    # Read the log file
    with open("../results/edge_connectivity_report.txt", "w") as f_report:
        for i in range(num_registrations):
            print(f"Edge dis-connectivity based on ip_registration{i:d}.log (running order):", file=f_report)
            with open(f"../results/ip_registration{i:d}.log", "r") as f:
                lines = f.readlines()
            # Extract the tile pair numbers from failed RANSAC correspondence finding log messages
            blocks = bigstitcher_log_edge_analysis.get_unfitted_tile_pairs(lines)
            # Create a visualization of the failed edges
            # Write the visualization to a file
            bigstitcher_log_edge_analysis.print_visualization(blocks, file=f_report)


def imagej_do_registrations(pipeline_manifest: ExaspimProcessingPipeline,
                            args: dict, logger: logging.Logger,
                            process_meta: DataProcess):  # pragma: no cover
    """Do the registrations.

    Do the registrations and create ng links for all the registrations and
    the original placements."""
    reg_index = 0
    if pipeline_manifest.ip_registrations:
        for ipreg_params in pipeline_manifest.ip_registrations:
            macro_reg = f"../results/macro_ip_reg{reg_index:d}.ijm"
            reg_params = ipreg_params.dict()
            reg_params.update(ipreg_params.IJwrap.dict())
            reg_params["process_xml"] = args["process_xml"]
            logger.info("Creating macro %s", macro_reg)
            with open(macro_reg, "w") as f:
                f.write(ImagejMacros.get_macro_ip_reg(reg_params))
            with open(f"../results/ip_registration{reg_index:d}.log", "w") as f_out:
                r = wrapper_cmd_run(
                    [
                        "ImageJ",
                        "-Dimagej.updater.disableAutocheck=true",
                        "--headless",
                        "--memory",
                        "{memgb}G".format(**reg_params),
                        "--console",
                        "--run",
                        macro_reg,
                    ],
                    logger,
                    f_stdout=f_out,
                    f_stderr=f_out,
                )
            if r != 0:
                raise RuntimeError(f"IP registration {reg_index} command failed.")
            reg_index += 1

        # Create ng links for all the registrations
        nglinks = []
        for i in range(reg_index + 1):
            suffix = f"~{i}" if i > 0 else ""
            xml_path = args["process_xml"] + suffix
            if os.path.exists(xml_path):
                logger.info("Creating ng link for registration %d (xml order)", i)
                thelink = create_ng_link(
                    "{}SPIM.ome.zarr".format(args["input_uri"]),
                    args["output_uri"].rstrip("/"),
                    xml_path=xml_path,
                    output_json=f"../results/ng/process_output_{i}.json",
                )
                if thelink:
                    nglinks.append(thelink)
            else:
                logger.warning("Registration %d xml file %s does not exist. Skipping.", i, xml_path)
        if process_meta.outputs is None:
            process_meta.outputs = {}
        if nglinks:
            process_meta.outputs["ng_links"] = nglinks
        logger.info("Creating edge connectivity report")
        create_edge_connectivity_report(reg_index)
    logger.info("Creating EMR ready xml from bigstitcher.xml")
    create_emr_ready_xml(args, num_regs=reg_index)


def imagej_wrapper_main():  # pragma: no cover
    """Entry point with the manifest config."""
    # logging.basicConfig(format="%(asctime)s %(name)s %(levelname)-7s %(message)s")
    logging.basicConfig(format="%(asctime)s %(levelname)-7s %(message)s")

    logger = logging.getLogger()
    pipeline_manifest = get_capsule_manifest()

    args = {
        "dataset_xml": "../data/manifest/dataset.xml",
        "session_id": pipeline_manifest.pipeline_suffix,
        "log_level": logging.DEBUG,
        "name": pipeline_manifest.name,
        "subject_id": pipeline_manifest.subject_id,
    }
    # "input_uri" and "output_uri" are formatted to have trailing slashes
    if pipeline_manifest.ip_registrations:
        args["output_uri"] = fmt_uri(pipeline_manifest.ip_registrations[-1].IJwrap.output_uri)
        args["input_uri"] = fmt_uri(pipeline_manifest.ip_registrations[-1].IJwrap.input_uri)
    else:
        args["output_uri"] = fmt_uri(pipeline_manifest.ip_detection.IJwrap.output_uri)
        args["input_uri"] = fmt_uri(pipeline_manifest.ip_detection.IJwrap.input_uri)

    logger.setLevel(logging.DEBUG)
    logging.getLogger("botocore").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.INFO)
    logging.getLogger("s3fs").setLevel(logging.INFO)

    args.update(get_auto_parameters(args))
    process_meta = get_imagej_wrapper_metadata(
        {
            "ip_detection": pipeline_manifest.ip_detection,
            "ip_registrations": pipeline_manifest.ip_registrations,
        },
        input_location=args["input_uri"],
        output_location=args["output_uri"],
    )

    write_process_metadata(process_meta, prefix="ipreg")
    ip_det_parameters = pipeline_manifest.ip_detection
    if ip_det_parameters is not None:
        logger.info("Copying input xml %s -> %s", args["dataset_xml"], args["process_xml"])
        shutil.copy(args["dataset_xml"], args["process_xml"])

        det_params = pipeline_manifest.ip_detection.dict()
        det_params.update(pipeline_manifest.ip_detection.IJwrap.dict())
        det_params["process_xml"] = args["process_xml"]
        logger.info("Creating macro %s", args["macro_ip_det"])
        with open(args["macro_ip_det"], "w") as f:
            f.write(ImagejMacros.get_macro_ip_det(det_params))
        with open("../results/ip_detection.log", "w") as f_out:
            r = wrapper_cmd_run(
                [
                    "ImageJ",
                    "-Dimagej.updater.disableAutocheck=true",
                    "--headless",
                    "--memory",
                    "{memgb}G".format(**det_params),
                    "--console",
                    "--run",
                    args["macro_ip_det"],
                ],
                logger,
                f_stdout=f_out,
                f_stderr=f_out,
            )
            if r != 0:
                raise RuntimeError("IP detection command failed.")
    else:
        if pipeline_manifest.ip_registrations:
            # At the moment we did not define an IP detection only dataset
            # We assume that interest point detections are already present in the input dataset
            # in the folder of the xml dataset file (in the manifest folder)
            logger.info("Assume already detected interestpoints.")
            ip_src = os.path.join(os.path.dirname(args["dataset_xml"]), "interestpoints.n5")
            logger.info("Copying %s -> ../results/", ip_src)
            shutil.copytree(ip_src, "../results/interestpoints.n5", dirs_exist_ok=True)

    # Separate function to keep overall complexity low
    imagej_do_registrations(pipeline_manifest, args, logger, process_meta)

    logger.info("Setting processing metadata to done")
    set_metadata_done(process_meta)
    write_process_metadata(process_meta, prefix="ipreg")
    logger.info("Uploading capsule results to {}".format(args["output_uri"]))
    upload_alignment_results(args)


if __name__ == "__main__":  # pragma: no cover
    main()

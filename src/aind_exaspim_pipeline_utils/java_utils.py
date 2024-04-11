"""Java IpdetRegSolv capsule postprocessing actions"""

import logging
import os

from .exaspim_manifest import get_capsule_manifest
from .imagej_wrapper import (
    create_edge_connectivity_report,
    upload_alignment_results,
    fmt_uri,
    get_auto_parameters,
)
from .qc.create_ng_link import create_ng_link


def java_detreg_postprocess_main():  # pragma: no cover
    """Entry point for java capsule postprocessing."""

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
    logger.info(f"This is result postprocessing for session {args['session_id']}")
    args.update(get_auto_parameters(args))

    # process_meta = get_imagej_wrapper_metadata(
    #     {
    #         "ip_detection": pipeline_manifest.ip_detection,
    #         "ip_registrations": pipeline_manifest.ip_registrations,
    #     },
    #     input_location=args["input_uri"],
    #     output_location=args["output_uri"],
    # )
    # write_process_metadata(process_meta, prefix="ipreg")

    # Create ng links for all the registrations
    nglinks = []
    for i in range(3):
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
    # if process_meta.outputs is None:
    #     process_meta.outputs = {}
    # if nglinks:
    #     process_meta.outputs["ng_links"] = nglinks
    logger.info("Creating edge connectivity report")
    create_edge_connectivity_report(2)

    logger.info("Uploading capsule results to {}".format(args["output_uri"]))
    upload_alignment_results(args)

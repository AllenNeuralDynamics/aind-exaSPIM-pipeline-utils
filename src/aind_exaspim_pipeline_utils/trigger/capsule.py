"""Trigger capsule actions"""

import argparse
import collections.abc
import datetime
import logging
import os
import time
import re
from typing import Optional, Union

import boto3
import json
from urllib.parse import urlparse
import urllib.request

import botocore.exceptions
from aind_codeocean_api.codeocean import CodeOceanClient
from aind_codeocean_api.models.computations_requests import ComputationDataAsset

from aind_trigger_codeocean.pipelines import CapsuleJob, RegisterAindData

from ..exaspim_manifest import (
    XMLCreationParameters,
    IJWrapperParameters,
    IPDetectionParameters,
    IPRegistrationParameters,
    ExaspimProcessingPipeline,
    N5toZarrParameters,
    ZarrMultiscaleParameters,
)

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
logger = logging.getLogger("exaspim_trigger")
logger.setLevel(logging.INFO)


def get_fname_timestamp(stamp: Optional[datetime.datetime] = None) -> str:
    """Get the time in the format used in the file names YYYY-MM-DD_HH-MM-SS"""
    if stamp is None:  # pragma: no cover
        stamp = datetime.datetime.now()
    return stamp.strftime("%Y-%m-%d_%H-%M-%S")


def parse_args() -> argparse.Namespace:  # pragma: no cover
    """Command line arguments of the trigger capsule.

    As we use CO "run" to specify these, all parameters can be empty string.
    """
    parser = argparse.ArgumentParser(
        prog="run_trigger_capsule",
        description="This program prepares the CO environment and launches the exaSPIM processing pipeline",
    )
    # parser.add_argument("--pipeline_id", help="CO pipeline id to launch")
    parser.add_argument(
        "--exaspim_data_uri",
        help="S3 URI Top-level location of input exaSPIM " "dataset in aind-open-data",
        required=True,
    )
    parser.add_argument(
        "--raw_data_uri",
        help="S3 URI Top-level location of input exaSPIM"
             "dataset in aind-open-data if different from exaspim_data_uri "
             "(ie. flat-fielded)",
    )
    parser.add_argument(
        "--manifest_output_prefix_uri", help="S3 prefix URI for processing manifest upload", required=True
    )
    parser.add_argument(
        "--pipeline_timestamp",
        help="Pipeline timestamp to be appended to folder names. "
             "Defaults to current local time as YYYY-MM-DD_HH-MM-SS",
    )
    parser.add_argument("--template_manifest", help="Template manifest json file to override defaults.")

    parser.add_argument("--xml_capsule_id", help="XML converter capsule id. Runs it if present.")
    parser.add_argument("--ij_capsule_id", help="ImageJ wrapper capsule id. Starts it if present.")
    parser.add_argument("--channel",
                        help="Channel name to process. Must be one of the wavelengths in the metadata. "
                             "Without the 'ch' prefix.")
    args = parser.parse_args()
    return args


def get_s3_file(bucket_name: str, object_name: str, local_file: str):  # pragma: no cover
    """Download a file from S3.

    Raises
    ------
    With the exception of 404, all other exceptions are raised.

    Returns
    -------

    Returns True if successful, False if not found.
    """
    s3 = boto3.client("s3")  # Authentication should be available in the environment
    logger.info(f"Downloading file s3://{bucket_name}/{object_name}")
    try:
        s3.download_file(bucket_name, object_name, local_file)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            logger.warning(f"File {object_name} not found in {bucket_name}. Skipping")
            return False
        else:
            raise
    return True


# TODO: Use validated model, once stable
def get_dataset_metadata(args) -> dict:  # pragma: no cover
    """Get the metadata jsons of the exaspim dataset from S3.

    Get `data_description`, `subject.json` and `acquisition.json` from the data location.

    Collect the contents of the individual jsons into top level keys
    like `acquisition` and `subject` and returns the dict.
    """
    metadata = {}

    # acquisiton and exaSPIM_acquisition must be the last two
    files = ["data_description.json", "subject.json", "acquisition.json", "exaSPIM_acquisition.json"]
    logger.info("Reading metadata files from S3.")
    for f in files:
        object_name = "/".join((args.input_dataset_prefix, f))
        # Try the input dataset
        if not get_s3_file(args.input_dataset_bucket_name, object_name, f"../results/{f}"):
            logger.warning(f"Metadata file {f} not found in {args.input_dataset_prefix}")
            object_name = "/".join((args.raw_dataset_prefix, f))
            if not get_s3_file(args.raw_dataset_bucket_name, object_name, f"../results/{f}"):
                logger.warning(f"Metadata file not found {f} in {args.raw_dataset_prefix}. Skipping.")
                continue

        with open(f"../results/{f}", "r") as jfile:
            data = json.load(jfile)
            metadata[os.path.splitext(f)[0]] = data

        # If acquisition is present in the input dataset,
        # don't try the old name in the raw dataset
        if "acquisition" in metadata:
            break

    # If acquisition has its old name, rename it
    if "exaSPIM_acquisition" in metadata:
        metadata["acquisition"] = metadata["exaSPIM_acquisition"]
        del metadata["exaSPIM_acquisition"]

    # Validate subject_id
    m = re.match(r".*exaSPIM_(\w+)_\d{4}-\d{2}-\d{2}", args.input_dataset_prefix)
    if m:
        fname_subject_id = m.group(1)
    else:
        raise ValueError("Cannot extract subject id from the input dataset prefix {}".format(
            args.input_dataset_prefix))

    if not metadata["subject"]:
        logger.warning("No subject metadata. Using file path pattern.")
        metadata["subject"] = {"subject_id": fname_subject_id}
    else:
        meta_subject_id = metadata["subject"].get("subject_id")
        if not meta_subject_id:
            logger.warning("Subject id is null in metadata. Using file path pattern.")
            metadata["subject"]["subject_id"] = fname_subject_id
        else:
            # We have an entry in the metadata
            if meta_subject_id != fname_subject_id:
                raise ValueError(
                    "The subject id in the metadata and in the file paths do not match. "
                    f"{meta_subject_id} != {fname_subject_id}"
                )

    return metadata


# TODO: Validate whether the location we're processing matches with the metadata given location
# def validate_s3_location(args, meta):  # pragma: no cover
#     """Get the last data_process and check whether we're at its output location"""
#     lastproc: DataProcess = DataProcess.parse_obj(meta["processing"]["data_processes"][-1])
#     meta_url = urlparse(lastproc.output_location)
#
#     if meta_url.netloc != args.input_dataset_bucket_name or \
#     meta_url.path.strip("/") != args.input_dataset_prefix:
#         raise ValueError(
#         "Output location of last DataProcess does not match with current dataset location")


def register_or_get_dataset_as_CO_data_asset(
        dataset_name, dataset_bucket, dataset_prefix, co_client
):  # pragma: no cover
    """Query the input dataset for existence and register it if not found.

    At the moment the dataset_name and dataset_prefix are the same in aind-open-data.
    """
    query_response = co_client.search_data_assets(query=f"name:{dataset_name}")
    if query_response.status_code != 200:
        raise RuntimeError("Cannot query data assets")
    query_response = query_response.json()
    results = query_response["results"]
    for r in results:
        if r["name"] == dataset_name:
            # Check that it points to the same bucket and prefix
            sourceBucket = r.get("sourceBucket", {})
            if sourceBucket.get("prefix") == dataset_prefix and sourceBucket.get("bucket") == dataset_bucket:
                logger.info(f"Found dataset {dataset_name} in CO as {r['id']}")
                return r["id"]
    # Not found, register
    logger.info(f"Register dataset as a data asset in CO. {dataset_bucket}:{dataset_name}")
    # Register data asset
    data_configs = {"prefix": dataset_prefix, "bucket": dataset_bucket}

    R = RegisterAindData(
        configs=data_configs, co_client=co_client, viewable_to_everyone=True, is_public_bucket=True
    )
    data_asset_reg_response = R.run_job()
    if data_asset_reg_response.status_code != 200:
        raise RuntimeError("Cannot register data asset")

    response_contents = data_asset_reg_response.json()
    logger.info(f"Created data asset {dataset_name} as {response_contents['id']}")
    return response_contents["id"]


def register_raw_dataset_as_CO_data_asset(args, meta, co_client):  # pragma: no cover
    """Register the raw dataset as a linked S3 data asset in CO.

    The raw dataset is different from the input dataset if the input dataset is flat-fielded.
    The alignment output inherits the raw dataset's timestamp."""

    logger.info(
        f"Register dataset as a data asset in CO. {args.raw_dataset_bucket_name} {args.raw_dataset_name}"
    )
    # Register data asset
    data_configs = {"prefix": args.raw_dataset_name, "bucket": args.raw_dataset_bucket_name}

    R = RegisterAindData(
        configs=data_configs, co_client=co_client, viewable_to_everyone=True, is_public_bucket=True
    )
    data_asset_reg_response = R.run_job()

    print(data_asset_reg_response)
    response_contents = data_asset_reg_response.json()
    logger.info(f"Created data asset in Code Ocean: {response_contents}")

    data_asset_id = response_contents["id"]

    return data_asset_id


class RegisterDataJob(CapsuleJob):  # pragma: no cover
    """Minimalistic object to run as a CapsuleJob"""

    def run_job(self):
        """An empty run_job method

        TODO: Move the actual job preparation and run code here.
        """
        pass


def register_manifest_as_CO_data_asset(args, co_client):  # pragma: no cover
    """Register the manifest as a linked S3 data asset in CO"""
    # TODO: Current metadata fails with schema validation
    # data_description: DataDescription = DataDescription.parse_obj(meta["data_description"])
    tags = ["exaSPIM", "manifest"]

    C = RegisterDataJob(configs={}, co_client=co_client)
    data_asset_reg_response = C.register_data(
        asset_name=args.manifest_name,
        mount="manifest",
        bucket=args.manifest_bucket_name,
        prefix=args.manifest_path,
        tags=tags,
        viewable_to_everyone=True,
        is_public_bucket=True,
    )

    response_contents = data_asset_reg_response.json()
    logger.info(f"Created data asset in Code Ocean: {response_contents}")

    data_asset_id = response_contents["id"]

    return data_asset_id


def start_pipeline(args, co_client, manifest_data_asset_id):  # pragma: no cover
    """Mount the manifest and start a CO pipeline or capsule."""
    # mount
    data_assets = [
        ComputationDataAsset(id=manifest_data_asset_id, mount="manifest"),
    ]
    C = RegisterDataJob(configs={}, co_client=co_client)
    run_response = C.run_capsule(capsule_id=args.ij_capsule_id, data_assets=data_assets)

    logger.info(f"Run response: {run_response.json()}")
    time.sleep(5)


def run_xml_capsule(args, co_client, input_data_asset_id, manifest_data_asset_id):  # pragma: no cover
    """Run the xml generator capsule.

    * Attach the input_data_asset_id as exaspim_dataset to the capsule
    * Run the capsule and waits for completion.
    * Download output.xml and upload it to the manifest location.
    """
    logger.info("Running xml creator capsule")
    data_assets = [
        ComputationDataAsset(id=input_data_asset_id, mount="exaspim_dataset"),
        ComputationDataAsset(id=manifest_data_asset_id, mount="manifest"),
    ]
    C = RegisterDataJob(configs={}, co_client=co_client)
    run_response = C.run_capsule(capsule_id=args.xml_capsule_id, data_assets=data_assets, pause_interval=10)

    run_response = run_response.json()

    result_response = co_client.get_result_file_download_url(run_response["id"], "output.xml")
    result = result_response.json()
    if result_response.status_code != 200 or "url" not in result:
        raise RuntimeError("Cannot get xml capsule result")
    logger.info(f"Result query response: {result}")
    urllib.request.urlretrieve(result["url"], "../results/dataset.xml")
    # Upload
    s3 = boto3.client("s3")  # Authentication should be available in the environment
    object_name = "/".join((args.manifest_path, "dataset.xml"))
    logger.info(f"Uploading to bucket {args.manifest_bucket_name} : {object_name}")
    s3.upload_file("../results/dataset.xml", args.manifest_bucket_name, object_name)


def start_ij_capsule(args, co_client, input_data_asset_id, manifest_data_asset_id):  # pragma: no cover
    """Start the IJ wrapper capsule."""
    logger.info(
        "Running IJ capsule with dataset {} and manifest {}".format(
            input_data_asset_id, manifest_data_asset_id
        )
    )
    data_assets = [
        ComputationDataAsset(id=input_data_asset_id, mount="exaspim_dataset"),
        ComputationDataAsset(id=manifest_data_asset_id, mount="manifest"),
    ]

    C = RegisterDataJob(configs={}, co_client=co_client)
    C.run_capsule(
        capsule_id=args.ij_capsule_id,
        data_assets=data_assets,
        pause_interval=30,
        timeout_seconds=3600 * 100,
    )
    logger.info("IJ capsule finished. Registering alignment dataset as a data asset in CO")
    # Create data asset from the output_uri location
    C.register_data(
        asset_name=args.alignment_dataset_name,
        mount=args.alignment_dataset_name,
        bucket=args.input_dataset_bucket_name,
        prefix=args.alignment_dataset_name,
        tags=["exaSPIM", "alignment"],
        viewable_to_everyone=True,
        is_public_bucket=True,
    )


def validate_channel_name(metadata: dict, args: argparse.Namespace) -> str:  # pragma: no cover
    """Validate the channel name in the metadata json and in the capsule arguments.

    If there is no channel name in the capsule arguments, set it in place.

    Returns
    -------

    The channel name is the nominal wavelength only, without the "ch" prefix.
    """
    ch_names = []
    if "acquisition" in metadata:
        acq = metadata["acquisition"]
        for t in acq["tiles"]:
            ch_names.append(t["channel"]["channel_name"])
    logger.info(f"Found channels in acquisition metadata: {set(ch_names)}")
    if args.channel:
        if args.channel not in ch_names:
            raise ValueError(
                f"Channel name {args.channel} not found in the metadata: {set(ch_names)}."
            )
    else:
        args.channel = ch_names[0]
    logger.info(f"Processing selected channel: {args.channel}")
    return args.channel


def recursive_assign_items(
        dst: Union[collections.abc.Mapping, list],
        key_dst: Union[str, int],
        src: Union[collections.abc.Mapping, list],
        key_src: Union[str, int],
) -> None:
    """Recursively assign items between dictionaries or lists.


    If src[key_src] is a dictionary, recursively assign items from src[key_src] to dst[key_dst].
    dst[key_dst] must exist.

    If src[key_src] is a list, assign all items recursively from src[key_src] to dst[key_dst].
    dst[key_dst] must exist and must be a list of the same length as src[key_src].

    For all other cases, dst[key_dst] = src[key_src] is executed.
    """
    if isinstance(src[key_src], collections.abc.Mapping):
        dst[key_dst] = recursive_update_mapping(dst[key_dst], src[key_src])
    elif isinstance(src[key_src], list):
        dlist = dst[key_dst]
        slist = src[key_src]
        if len(dlist) != len(slist):
            raise ValueError("List field lengths to assign do not match")
        for i in range(len(slist)):
            recursive_assign_items(dlist, i, slist, i)
    else:
        dst[key_dst] = src[key_src]


def recursive_update_mapping(
        dst: collections.abc.Mapping, src: collections.abc.Mapping
) -> collections.abc.Mapping:
    """Recursively update a dictionary-like object.

    Use to override items in a parameter configuration dictionary (manifest). The overriding dictionary
    shall have the same hierarchy as the original one but only contains the entries to override.

    If a list item is to be updated, the overriding dictionary shall contain a list of the same length and
    oll the not to be updated dictionary items of the list can be empty.

    The hierarchy is traversed following ``src`` and ``dst`` is expected to have all encountered keys.
    """
    for key in src:
        recursive_assign_items(dst, key, src, key)
    return dst


def create_exaspim_manifest(args, metadata):  # pragma: no cover
    """Create exaspim manifest from the dataset metadata that we have.

    If args.template_manifest is present, override the defaults with it."""
    # capsule_xml_path = "../data/manifest/dataset.xml"
    def_ij_wrapper_parameters: IJWrapperParameters = IJWrapperParameters(
        memgb=106, parallel=32, input_uri=args.exaspim_data_uri, output_uri=args.alignment_output_uri
    )

    def_ip_detection_parameters: IPDetectionParameters = IPDetectionParameters(
        # dataset_xml=capsule_xml_path,  # For future S3 path
        IJwrap=def_ij_wrapper_parameters,
        downsample=4,
        bead_choice="manual",
        sigma=1.8,
        threshold=0.03,
        find_minima=False,
        find_maxima=True,
        set_minimum_maximum=True,
        minimal_intensity=0,
        maximal_intensity=2000,
        ip_limitation_choice="brightest",
        maximum_number_of_detections=150000,
    )
    ip_reg_translation: IPRegistrationParameters = IPRegistrationParameters(
        # dataset_xml=capsule_xml_path,
        IJwrap=def_ij_wrapper_parameters,
        transformation_choice="translation",
        compare_views_choice="overlapping_views",
        interest_point_inclusion_choice="overlapping_ips",
        fix_views_choice="select_fixed",
        fixed_tile_ids=(7,),
        map_back_views_choice="no_mapback",
        do_regularize=False,
    )
    ip_reg_affine: IPRegistrationParameters = IPRegistrationParameters(
        # dataset_xml=capsule_xml_path,
        IJwrap=def_ij_wrapper_parameters,
        transformation_choice="affine",
        compare_views_choice="overlapping_views",
        interest_point_inclusion_choice="overlapping_ips",
        fix_views_choice="select_fixed",
        fixed_tile_ids=(7,),
        map_back_views_choice="no_mapback",
        do_regularize=True,
        regularize_with_choice="rigid",
    )

    ch_name = args.channel
    # Even the flat-fielded fusions goes with the raw dataset prefix
    n5_to_zarr: N5toZarrParameters = N5toZarrParameters(
        voxel_size_zyx=(1.0, 0.748, 0.748),
        input_uri=f"s3://{args.fusion_output_bucket}/{args.fusion_output_prefix}/fused.n5/ch{ch_name}/",
        output_uri=f"s3://{args.fusion_output_bucket}/{args.fusion_output_prefix}/fused.zarr/",
    )

    zarr_multiscale: ZarrMultiscaleParameters = ZarrMultiscaleParameters(
        voxel_size_zyx=(1.0, 0.748, 0.748),
        input_uri=f"s3://{args.fusion_output_bucket}/{args.fusion_output_prefix}/fused.zarr/",
    )

    xml_creation: XMLCreationParameters = XMLCreationParameters(ch_name=ch_name)

    processing_manifest: ExaspimProcessingPipeline = ExaspimProcessingPipeline(
        creation_time=args.pipeline_timestamp,
        pipeline_suffix=args.fname_timestamp,
        subject_id=metadata["subject"].get("subject_id"),
        name=metadata["data_description"].get("name"),
        xml_creation=xml_creation,
        ip_detection=def_ip_detection_parameters,
        ip_registrations=[ip_reg_translation, ip_reg_affine],
        n5_to_zarr=n5_to_zarr,
        zarr_multiscale=zarr_multiscale,
    )
    if args.template_manifest:
        logger.info(f"Overriding manifest entries from {args.template_manifest}")
        with open(args.template_manifest, "r") as f:
            template = json.load(f)
        processing_manifest = ExaspimProcessingPipeline(
            **recursive_update_mapping(processing_manifest.dict(), template)
        )

    return processing_manifest


def create_and_upload_emr_config(args, manifest: ExaspimProcessingPipeline):  # pragma: no cover
    """Create EMR command line parameters for the fusion of the present alignment run."""
    ch_name = args.channel
    config = (
        f"-x, {args.alignment_output_uri}"
        f"bigstitcher_emr_{manifest.subject_id}_{manifest.pipeline_suffix}_0.xml,\n"
        f"--outS3Bucket, {args.fusion_output_bucket}, -o, /{args.fusion_output_prefix}/fused.n5,\n"
        f"-d, /ch{ch_name}/s0, --storage, N5, --UINT16, --minIntensity=0, "
        f"--maxIntensity=65535, --preserveAnisotropy\n"
    )
    with open("../results/emr_fusion_config.txt", "w") as f:
        f.write(config)
    logger.info("Uploading emr_fusion_config.txt to bucket {}".format(args.manifest_bucket_name))
    s3 = boto3.client("s3")  # Authentication should be available in the environment
    object_name = "/".join((args.manifest_path, "emr_fusion_config.txt"))
    s3.upload_file("../results/emr_fusion_config.txt", args.manifest_bucket_name, object_name)


def upload_manifest(args, manifest: ExaspimProcessingPipeline):  # pragma: no cover
    """Write out the given manifest as a json file and upload to S3"""
    s3 = boto3.client("s3")  # Authentication should be available in the environment
    object_name = "/".join((args.manifest_path, "exaspim_manifest.json"))
    with open("../results/exaspim_manifest.json", "w") as f:
        f.write(manifest.json(indent=4))
    logger.info(f"Uploading manifest to bucket {args.manifest_bucket_name} : {object_name}")
    s3.upload_file("../results/exaspim_manifest.json", args.manifest_bucket_name, object_name)


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


def process_args(args):  # pragma: no cover
    """Command line arguments processing"""

    # Determine the pipeline timestamp
    if args.pipeline_timestamp is None:
        pipeline_timestamp = datetime.datetime.now()
    else:
        pipeline_timestamp = datetime.datetime.strptime(args.pipeline_timestamp, "%Y-%m-%d_%H-%M-%S")

    args.pipeline_timestamp = pipeline_timestamp
    args.fname_timestamp = get_fname_timestamp(pipeline_timestamp)
    args.exaspim_data_uri = fmt_uri(args.exaspim_data_uri)

    # Get raw dataset bucket and path
    url = urlparse(args.exaspim_data_uri)
    args.input_dataset_bucket_name = url.netloc
    # Includes the last element and optionally other path elements
    # No slashes at the beginning and end of prefixes
    args.input_dataset_prefix = url.path.strip("/")
    args.input_dataset_name = os.path.basename(args.input_dataset_prefix)  # Only the last entry as "name"
    if args.raw_data_uri:
        # There is a separate raw dataset given - the input dataset is flat-fielded
        args.raw_data_uri = fmt_uri(args.raw_data_uri)
        url = urlparse(args.raw_data_uri)
        args.raw_dataset_bucket_name = url.netloc
        args.raw_dataset_prefix = url.path.strip("/")  # The path including the raw dataset name
        args.raw_dataset_name = os.path.basename(args.raw_dataset_prefix)  # Only the last entry as "name"
    else:
        # The input dataset is a raw dataset
        args.raw_dataset_bucket_name = args.input_dataset_bucket_name
        args.raw_dataset_prefix = args.input_dataset_prefix
        args.raw_dataset_name = args.input_dataset_name
    # Get manifest bucket and path and 'directory' name
    url = urlparse(args.manifest_output_prefix_uri)
    args.manifest_bucket_name = url.netloc
    manifest_name = "exaspim_manifest_{}".format(args.fname_timestamp)
    # S3 "directory" path for uploading generated manifest file
    args.manifest_name = manifest_name
    args.manifest_path = url.path.strip("/") + "/" + manifest_name
    # Alignment result upload location
    args.alignment_dataset_name = "{}_alignment_{}".format(args.raw_dataset_name, args.fname_timestamp)
    args.alignment_output_uri = "s3://{}/{}/".format(
        args.input_dataset_bucket_name, args.alignment_dataset_name
    )
    args.fusion_output_bucket = args.input_dataset_bucket_name
    args.fusion_output_prefix = "{}_fusion_{}".format(args.raw_dataset_prefix, args.fname_timestamp)


def capsule_main():  # pragma: no cover
    """Main entry point for trigger capsule."""

    args = parse_args()  # To get help before the error messages
    cwd = os.getcwd()
    if os.path.basename(cwd) != "code":
        # We don't know where we are in the capsule environment
        raise RuntimeError("This program must be run from the 'code' capsule folder.")

    if "CODEOCEAN_DOMAIN" not in os.environ or "CUSTOM_KEY" not in os.environ:
        raise RuntimeError(
            "CODEOCEAN_DOMAIN and CUSTOM_KEY variables must be set with CO API access credentials"
        )

    process_args(args)
    logger.info("This is pipeline run {}".format(args.fname_timestamp))
    metadata = get_dataset_metadata(args)
    validate_channel_name(metadata, args)
    # Creating the API Client
    co_client = CodeOceanClient(domain=os.environ["CODEOCEAN_DOMAIN"], token=os.environ["CUSTOM_KEY"])
    # validate_s3_location(args, metadata)
    input_data_asset_id = register_or_get_dataset_as_CO_data_asset(
        args.input_dataset_name, args.input_dataset_bucket_name, args.input_dataset_prefix, co_client
    )

    if args.raw_data_uri and args.raw_dataset_prefix != args.input_dataset_prefix:
        register_or_get_dataset_as_CO_data_asset(
            args.raw_dataset_name, args.raw_dataset_bucket_name, args.raw_dataset_prefix, co_client
        )
    manifest = create_exaspim_manifest(args, metadata)
    upload_manifest(args, manifest)
    create_and_upload_emr_config(args, manifest)

    # The XML also goes into this but we need the manifest now. CO index may miss the xml
    manifest_data_asset_id = register_manifest_as_CO_data_asset(args, co_client)
    if args.xml_capsule_id:
        run_xml_capsule(args, co_client, input_data_asset_id, manifest_data_asset_id)
    if args.ij_capsule_id:
        start_ij_capsule(args, co_client, input_data_asset_id, manifest_data_asset_id)
    logger.info("Done.")

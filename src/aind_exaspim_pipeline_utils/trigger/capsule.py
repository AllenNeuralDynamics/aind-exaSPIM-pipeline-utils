"""Trigger capsule actions"""

import argparse
import datetime
import os
import time
from typing import Optional

import boto3
import json
from urllib.parse import urlparse
import urllib.request

from aind_codeocean_api.codeocean import CodeOceanClient
from aind_data_schema import DataProcess

from ..exaspim_manifest import (
    IJWrapperParameters,
    IPDetectionParameters,
    IPRegistrationParameters,
    ExaspimProcessingPipeline,
    N5toZarrParameters,
    ZarrMultiscaleParameters,
)


def get_fname_timestamp(stamp: Optional[datetime.datetime] = None) -> str:  # pragma: no cover
    """Get the time in the format used in the file names YYYY-MM-DD_HH-MM-SS"""
    if stamp is None:
        stamp = datetime.datetime.utcnow()
    return stamp.strftime("%Y-%m-%d_%H-%M-%S")


def parse_args() -> argparse.Namespace:  # pragma: no cover
    """Command line arguments of the trigger capsule"""
    parser = argparse.ArgumentParser(
        prog="run_trigger_capsule",
        description="This program prepares the CO environment and launches the exaSPIM processing pipeline",
    )
    parser.add_argument("--pipeline_id", help="CO pipeline id to launch")
    parser.add_argument(
        "exaspim_data_uri", help="S3 URI Top-level location of exaSPIM " "dataset in aind-open-data"
    )
    parser.add_argument("manifest_prefix_uri", help="S3 prefix URI for processing manifest upload")
    parser.add_argument("--pipeline_timestamp", help="Pipeline timestamp to be appended to folder names. "
                                                     "Defaults to current UTC time as YYYY-MM-DD_HH-MM-SS")
    parser.add_argument("--xml_capsule_id", help="XML converter capsule id. Runs it if present.")
    parser.add_argument("--ij_capsule_id", help="ImageJ wrapper capsule id. Starts it if present.")
    args = parser.parse_args()
    return args


# TODO: Use validated model, once stable
def get_dataset_metadata(args) -> dict:  # pragma: no cover
    """Get the metadata of the exaspim dataset from S3.

    Also gets the acquisition.json if available and puts into the dict.
    """

    s3 = boto3.client("s3")  # Authentication should be available in the environment

    object_name = "/".join((args.dataset_prefix, "metadata.json"))

    # Download the file from S3
    print(f"Downloading from bucket {args.dataset_bucket_name} : {object_name}")
    s3.download_file(args.dataset_bucket_name, object_name, "../results/metadata.json")

    # Parse the JSON file
    with open("../results/metadata.json", "r") as f:
        metadata = json.load(f)

    object_name = "/".join((args.dataset_prefix, "acquisition.json"))
    # Download the file from S3
    print(f"Downloading from bucket {args.dataset_bucket_name} : {object_name}")
    s3.download_file(args.dataset_bucket_name, object_name, "../results/acquisition.json")

    if os.path.exists("../results/acquisition.json"):
        with open("../results/acquisition.json", "r") as f:
            acqdata = json.load(f)
        if acqdata:
            metadata['acquisition'] = acqdata

    return metadata


def validate_s3_location(args, meta):  # pragma: no cover
    """Get the last data_process and check whether we're at its output location"""
    lastproc: DataProcess = DataProcess.parse_obj(meta["processing"]["data_processes"][-1])
    meta_url = urlparse(lastproc.output_location)

    if meta_url.netloc != args.dataset_bucket_name or meta_url.path.strip("/") != args.dataset_prefix:
        raise ValueError("Output location of last DataProcess does not match with current dataset location")


def wait_for_data_availability(
        co_client,
        data_asset_id: str,
        timeout_seconds: int = 300,
        pause_interval=10,
):  # pragma: no cover
    """
    There is a lag between when a register data request is made and when the
    data is available to be used in a capsule.
    Parameters
    ----------
    data_asset_id : str
    timeout_seconds : int
        Roughly how long the method should check if the data is available.
    pause_interval : int
        How many seconds between when the backend is queried.

    Returns
    -------
    requests.Response

    """
    num_of_checks = 0
    break_flag = False
    time.sleep(pause_interval)
    response = co_client.get_data_asset(data_asset_id)

    if ((pause_interval * num_of_checks) > timeout_seconds) or (response.status_code == 200):
        break_flag = True
    while not break_flag:
        print("Data asset is not yet available")
        print(response)
        time.sleep(pause_interval)
        response = co_client.get_data_asset(data_asset_id)
        num_of_checks += 1
        if ((pause_interval * num_of_checks) > timeout_seconds) or (response.status_code == 200):
            break_flag = True
    return response


def wait_for_compute_completion(
        co_api,
        compute_id: str,
        timeout_seconds: int = 300,
        pause_interval: int = 5,
):  # pragma: no cover
    """
    Parameters
    ----------
    data_asset_id : str
    timeout_seconds : int
        Roughly how long the method should check if the data is available.
    pause_interval : int
        How many seconds between when the backend is queried.

    Returns
    -------
    run_status: dict
        last run_status as json dict.

    """
    for i_check in range(timeout_seconds // pause_interval + 2):
        time.sleep(pause_interval)
        run_status = co_api.get_computation(compute_id)
        if run_status.status_code != 200:
            raise RuntimeError(f"Cannot get compute status {compute_id}")
        run_status = run_status.json()
        if run_status['state'] == 'completed' and run_status['has_results'] and \
                run_status['end_status'] == 'succeeded':
            break
        print(f"Waiting loop {i_check}: {run_status}")
    else:
        raise RuntimeError(f"Wait for {compute_id} timed out or ended unsuccessfully.")
    return run_status


def make_data_viewable(co_client: CodeOceanClient, data_asset_id: str):  # pragma: no cover
    """
    Makes a registered dataset viewable

    Parameters
    ----------
    co_client: CodeOceanClient
        Code ocean client

    """
    response_data_available = wait_for_data_availability(co_client, data_asset_id)

    if response_data_available.status_code != 200:
        raise FileNotFoundError(f"Unable to find: {data_asset_id}")

    # Make data asset viewable to everyone
    update_data_perm_response = co_client.update_permissions(data_asset_id=data_asset_id, everyone="viewer")
    print(f"Data asset viewable to everyone: {update_data_perm_response}")


def register_raw_dataset_as_CO_data_asset(args, meta, co_api):  # pragma: no cover
    """Register the dataset as a linked S3 data asset in CO"""
    # TODO: Current metadata fails with schema validation
    # data_description: DataDescription = DataDescription.parse_obj(meta["data_description"])
    tags = ["exaspim", "raw"]

    # TODO: Newer version of co_api supports custom metadata entries. Fill from metadata/data_description
    # Registering data asset
    data_asset_reg_response = co_api.register_data_asset(
        asset_name=args.dataset_name,
        mount=args.dataset_name,
        bucket=args.dataset_bucket_name,
        prefix=args.dataset_prefix,
        tags=tags,
    )
    print(data_asset_reg_response)
    response_contents = data_asset_reg_response.json()
    print(f"Created data asset in Code Ocean: {response_contents}")

    data_asset_id = response_contents["id"]
    # Making the created data asset available for everyone
    make_data_viewable(co_api, data_asset_id)

    return data_asset_id


def register_manifest_as_CO_data_asset(args, co_api):  # pragma: no cover
    """Register the manifest as a linked S3 data asset in CO"""
    # TODO: Current metadata fails with schema validation
    # data_description: DataDescription = DataDescription.parse_obj(meta["data_description"])
    tags = ["exaspim", "manifest"]

    # TODO: Newer version of co_api supports custom metadata entries. Fill from metadata/data_description
    # Registering data asset
    data_asset_reg_response = co_api.register_data_asset(
        asset_name=args.manifest_name,
        mount="manifest",
        bucket=args.manifest_bucket_name,
        prefix=args.manifest_path,
        tags=tags,
    )
    response_contents = data_asset_reg_response.json()
    print(f"Created data asset in Code Ocean: {response_contents}")

    data_asset_id = response_contents["id"]
    # Making the created data asset available for everyone
    make_data_viewable(co_api, data_asset_id)

    return data_asset_id


def start_pipeline(args, co_api, manifest_data_asset_id):  # pragma: no cover
    """Mount the manifest and start a CO pipeline or capsule."""
    # mount
    data_assets = [
        {"id": manifest_data_asset_id, "mount": "manifest"},
    ]

    # dumped_parameters = json.dumps(job_configs)
    # print(dumped_parameters)
    # Executing capsule attaching new data asset
    run_response = co_api.run_capsule(
        capsule_id=args.pipeline_id,
        data_assets=data_assets,
        parameters=None,
    )

    print(f"Run response: {run_response.json()}")
    time.sleep(5)


def run_xml_capsule(args, co_api, raw_data_asset_id):  # pragma: no cover
    """Run the xml generator capsule.

      * Attach the raw_data_asset_id as exaspim_dataset to the capsule
      * Run the capsule and waits for completion.
      * Download output.xml and upload it to the manifest location.
    """
    data_assets = [
        {"id": raw_data_asset_id, "mount": "exaspim_dataset"},
    ]

    # dumped_parameters = json.dumps(job_configs)
    # print(dumped_parameters)
    # Executing capsule attaching new data asset
    run_response = co_api.run_capsule(
        capsule_id=args.xml_capsule_id,
        data_assets=data_assets,
        parameters=None,
    )

    run_response = run_response.json()
    compute_id = run_response["id"]

    print(f"Run response: {run_response}")
    wait_for_compute_completion(co_api, compute_id)

    result_response = co_api.get_result_file_download_url(run_response["id"], "output.xml")
    result = result_response.json()
    if result_response.status_code != 200 or 'url' not in result:
        raise RuntimeError("Cannot get xml capsule result")
    print(f"Result query response: {result}")
    urllib.request.urlretrieve(result['url'], "../results/dataset.xml")
    # Upload
    s3 = boto3.client("s3")  # Authentication should be available in the environment
    object_name = "/".join((args.manifest_path, "dataset.xml"))
    print(f"Uploading to bucket {args.manifest_bucket_name} : {object_name}")
    s3.upload_file("../results/dataset.xml", args.manifest_bucket_name, object_name)


def start_ij_capsule(args, co_api, raw_data_asset_id, manifest_data_asset_id):  # pragma: no cover
    """Start the IJ wrapper capsule.
    """
    data_assets = [
        {"id": raw_data_asset_id, "mount": "exaspim_dataset"},
        {"id": manifest_data_asset_id, "mount": "manifest"}
    ]

    print("Starting IJ wrapper capsule")
    run_response = co_api.run_capsule(
        capsule_id=args.ij_capsule_id,
        data_assets=data_assets,
        parameters=None,
    )

    run_response = run_response.json()
    print(f"Run response: {run_response}")


def get_channel_name(metadata: dict):  # pragma: no cover
    """Get the channel name from the metadata json"""
    if 'acquisition' in metadata:
        acq = metadata['acquisition']
        ch_name = acq["tiles"][0]["channel"]["channel_name"]
    else:
        print("Warning: Cannot get channel name, defaults to ch488")
        ch_name = "ch488"
    return ch_name


def create_exaspim_manifest(args, metadata):  # pragma: no cover
    """Create exaspim manifest from the metadata that we have"""
    # capsule_xml_path = "../data/manifest/dataset.xml"
    def_ij_wrapper_parameters: IJWrapperParameters = IJWrapperParameters(memgb=106, parallel=32)
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

    ch_name = get_channel_name(metadata)
    # TODO: Generate plausible URIs
    n5_to_zarr: N5toZarrParameters = N5toZarrParameters(
        voxel_size_zyx=(1.0, 0.748, 0.748),
        input_uri=f"s3://{args.dataset_bucket_name}/{args.dataset_prefix}"
                  f"_fusion_{args.fname_timestamp}/fused.n5/{ch_name}/",
        output_uri=f"s3://{args.dataset_bucket_name}/{args.dataset_prefix}"
                   f"_fusion_{args.fname_timestamp}/fused.zarr/",
    )

    zarr_multiscale: ZarrMultiscaleParameters = ZarrMultiscaleParameters(
        voxel_size_zyx=(1.0, 0.748, 0.748),
        input_uri=f"s3://{args.dataset_bucket_name}/{args.dataset_prefix}"
                  f"_fusion_{args.fname_timestamp}/fused.zarr/",
    )

    processing_manifest: ExaspimProcessingPipeline = ExaspimProcessingPipeline(
        creation_time=args.pipeline_timestamp,
        pipeline_suffix=args.fname_timestamp,
        name=metadata["data_description"].get("name"),
        ip_detection=def_ip_detection_parameters,
        ip_registrations=[ip_reg_translation, ip_reg_affine],
        n5_to_zarr=n5_to_zarr,
        zarr_multiscale=zarr_multiscale,
    )

    return processing_manifest


def upload_manifest(args, manifest: ExaspimProcessingPipeline):  # pragma: no cover
    """Write out the given manifest as a json file and upload to S3"""
    s3 = boto3.client("s3")  # Authentication should be available in the environment
    object_name = "/".join((args.manifest_path, "exaspim_manifest.json"))
    with open("../results/exaspim_manifest.json", "w") as f:
        f.write(manifest.json(indent=4))
    print(f"Uploading manifest to bucket {args.manifest_bucket_name} : {object_name}")
    s3.upload_file("../results/exaspim_manifest.json", args.manifest_bucket_name, object_name)


def process_args(args):  # pragma: no cover
    """Command line arguments processing"""

    # Determine the pipeline timestamp
    if args.pipeline_timestamp is None:
        pipeline_timestamp = datetime.datetime.utcnow()
    else:
        pipeline_timestamp = datetime.datetime.strptime(args.pipeline_timestamp, "%Y-%m-%d_%H-%M-%S")

    args.pipeline_timestamp = pipeline_timestamp
    args.fname_timestamp = get_fname_timestamp(pipeline_timestamp)

    # Get raw dataset bucket and path
    url = urlparse(args.exaspim_data_uri)
    args.dataset_bucket_name = url.netloc
    # Includes the last element and optionally other path elements
    # No slashes at the beginning and end
    args.dataset_prefix = url.path.strip("/")
    args.dataset_name = os.path.basename(args.dataset_prefix)  # Only the last entry as "name"
    # Get manifest bucket and path and 'directory' name
    url = urlparse(args.manifest_prefix_uri)
    args.manifest_bucket_name = url.netloc
    manifest_name = "exaspim_manifest_{}".format(args.fname_timestamp)
    # S3 "directory" path for uploading generated manifest file
    args.manifest_name = manifest_name
    args.manifest_path = url.path.strip("/") + "/" + manifest_name


def capsule_main():  # pragma: no cover
    """Main entry point for trigger capsule."""
    args = parse_args()  # To get help before the error messages

    cwd = os.getcwd()
    if os.path.basename(cwd) != "code":
        # We don't know where we are in the capsule environment
        raise RuntimeError("This program should be run from the 'code' capsule folder.")

    if "CODEOCEAN_DOMAIN" not in os.environ or "CUSTOM_KEY" not in os.environ:
        raise RuntimeError(
            "CODEOCEAN_DOMAIN and CUSTOM_KEY variables must be set with CO API access credentials"
        )

    process_args(args)
    metadata = get_dataset_metadata(args)

    # Get code ocean creds
    # co_cred = CodeOceanCredentials(
    #     domain=os.environ["CODEOCEAN_DOMAIN"], token=os.environ["CUSTOM_KEY"]
    # ).dict()

    # Creating the API Client
    co_api = CodeOceanClient(domain=os.environ["CODEOCEAN_DOMAIN"], token=os.environ["CUSTOM_KEY"])
    # validate_s3_location(args, metadata)
    raw_data_asset_id = register_raw_dataset_as_CO_data_asset(args, metadata, co_api)
    manifest = create_exaspim_manifest(args, metadata)
    upload_manifest(args, manifest)
    manifest_data_asset_id = register_manifest_as_CO_data_asset(args, co_api)
    if args.xml_capsule_id:
        run_xml_capsule(args, co_api, raw_data_asset_id)
    if args.ij_capsule_id:
        start_ij_capsule(args, co_api, raw_data_asset_id, manifest_data_asset_id)

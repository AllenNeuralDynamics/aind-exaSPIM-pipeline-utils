"""Manifest declaration for the exaSPIM capsules"""
import os
import sys
from datetime import datetime
from typing import Optional, Tuple

from aind_data_schema import DataProcess, Processing
from aind_data_schema.base import AindModel
from aind_data_schema.data_description import Institution
from aind_data_transfer.util import file_utils
from pydantic import Field


# Based on aind-data-transfer/scripts/processing_manifest.py


class DatasetStatus(AindModel):  # pragma: no cover
    """Status of the datasets to control next processing step. TBD"""

    # status: Status = Field(
    #     ...,
    #     description="Status of the dataset on the local storage",
    #     title="Institution",
    #     enumNames=[i.value for i in Status],
    # )
    # Creating datetime
    status_date = Field(
        datetime.now().date().strftime("%Y-%m-%d"),
        title="Date the flag was created",
    )
    status_time = Field(
        datetime.now().time().strftime("%H-%M-%S"),
        title="Time the flag was created",
    )


class N5toZarrParameters(AindModel):  # pragma: no cover
    """N5 to zarr conversion configuration parameters.

    n5tozarr_da_converter Code Ocean task config parameters."""

    voxel_size_zyx: Tuple[float, float, float] = Field(
        ..., title="Z,Y,X voxel size in micrometers for output metadata"
    )

    input_uri: str = Field(
        ...,
        title="Input N5 dataset path. Must be a local filesystem path or "
        "start with s3:// to trigger S3 direct access.",
    )

    output_uri: str = Field(
        ...,
        title="Output Zarr dataset path. Must be a local filesystem path or "
        "start with s3:// to trigger S3 direct access. "
        "Must be different from the input_uri. Will be overwritten if exists.",
    )


class ZarrMultiscaleParameters(AindModel):  # pragma: no cover
    """N5 to zarr conversion configuration parameters.

    zarr_multiscale Code Ocean task config parameters."""

    voxel_size_zyx: Tuple[float, float, float] = Field(
        ..., title="Z,Y,X voxel size in micrometers for output metadata"
    )

    input_uri: str = Field(
        ...,
        title="Input Zarr group dataset path. Must be a local filesystem path or "
        "start with s3:// to trigger S3 direct access.",
    )

    output_uri: Optional[str] = Field(None, title="Output Zarr group dataset path if different from input.")


class ExaspimProcessingPipeline(AindModel):  # pragma: no cover
    """ExaSPIM processing pipeline configuration parameters

    If a field is None, it is considered to be a disabled step."""

    n5_to_zarr: N5toZarrParameters = Field(None, title="N5 to single scale Zarr conversion")
    zarr_multiscale: ZarrMultiscaleParameters = Field(None, title="Zarr to multiscale Zarr conversion")


class ExaspimManifest(AindModel):  # pragma: no cover
    """Manifest definition of an exaSPIM processing session.

    Connects the dataset and its pipeline processing history."""

    schema_version: str = Field("0.1.0", title="Schema Version", const=True)
    license: str = Field("CC-BY-4.0", title="License", const=True)

    specimen_id: str = Field(..., title="Specimen ID")
    # dataset_status: DatasetStatus = Field(
    #     ..., title="Dataset status", description="Dataset status"
    # )
    institution: Institution = Field(
        ...,
        description="An established society, corporation, foundation or other organization "
        "that collected this data",
        title="Institution",
        enumNames=[i.value.name for i in Institution],
    )
    # acquisition: Acquisition = Field(
    #     ...,
    #     title="Acquisition data",
    #     description="Acquition data coming from the rig which is necessary to create matadata files",
    # )

    processing_pipeline: ExaspimProcessingPipeline = Field(
        ...,
        title="ExaSPIM pipeline parameters",
        description="Parameters necessary for the exaspim pipeline steps.",
    )
    processing: Processing = Field(
        ...,
        title="ExaSPIM pipeline processing steps log",
        description="Processing steps that has already taken place.",
    )


def create_example_manifest(printit=True) -> ExaspimManifest:  # pragma: no cover
    """Create example manifest file

    Parameters
    ----------
    printit: bool
      Print the example?

    Returns
    -------
    example_manifest: ExaspimManifest
    """
    # print(ProcessingManifest.schema_json(indent=2))
    # print(ProcessingManifest.schema())

    processing_manifest_example = ExaspimManifest(
        specimen_id="653431",
        institution=Institution.AIND,
        processing_pipeline=ExaspimProcessingPipeline(
            n5_to_zarr=N5toZarrParameters(
                voxel_size_zyx=(1.0, 0.748, 0.748),
                input_uri="s3://aind-scratch-data/gabor.kovacs/2023-07-25_1653_BSS_fusion_653431/ch561/",
                output_uri="s3://aind-scratch-data/gabor.kovacs/n5_to_zarr_CO_2023-08-17_1351/",
            ),
            zarr_multiscale=ZarrMultiscaleParameters(
                voxel_size_zyx=(1.0, 0.748, 0.748),
                input_uri="s3://aind-scratch-data/gabor.kovacs/2023-07-25_1653_BSS_fusion_653431/ch561/",
            ),
        ),
        processing=Processing(data_processes=[]),
    )

    if printit:
        print(processing_manifest_example.json(indent=3))
        return  # If printed, we assume call from the cli
    return processing_manifest_example


def get_capsule_manifest():  # pragma: no cover
    """Get the manifest file from its Code Ocean location or as given in the cmd-line argument.

    Raises
    ------
    If the manifest is not found, required fields will be missing at schema validation.
    """
    if len(sys.argv) > 1:
        manifest_name = sys.argv[1]
    else:
        manifest_name = "data/manifest/exaspim_manifest.json"
    json_data = file_utils.read_json_as_dict(manifest_name)
    return ExaspimManifest(**json_data)


def append_metadata_to_manifest(capsule_manifest: ExaspimManifest, process: DataProcess) -> None:
    """Append the given dataprocess metadata to the exaspim pipeline manifest

    So long the pipeline is a linear sequence of steps, this should always be the
    case.
    """
    capsule_manifest.processing.data_processes.append(process)


def write_result_manifest(capsule_manifest: ExaspimManifest) -> None:
    """Write the updated manifest file to the Code Ocean results folder."""
    os.makedirs("results/manifest", exist_ok=True)
    with open("results/manifest/exaspim_manifest.json", "w") as f:
        f.write(capsule_manifest.json(indent=3))


def write_result_metadata(capsule_metadata: DataProcess) -> None:
    """Write the metadata file to the Code Ocean results folder."""
    os.makedirs("results/meta", exist_ok=True)
    with open("results/meta/exaspim_process.json", "w") as f:
        f.write(capsule_metadata.json(indent=3))


if __name__ == "__main__":
    create_example_manifest(printit=True)

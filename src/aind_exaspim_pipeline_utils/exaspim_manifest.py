"""Manifest declaration for the exaSPIM capsules"""
import sys
from datetime import datetime
from typing import Optional, Tuple

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

    input_bucket: Optional[str] = Field(
        None, title="The input bucket. If specified, triggers reading from S3 directly."
    )

    input_name: str = Field(
        ...,
        title="Input N5 dataset path. Interpreted as relative to the bucket if given, otherwise,"
        "as a path on the local filesystem.",
    )

    output_bucket: Optional[str] = Field(
        None, title="The output S3 bucket. If sepcified, triggers writing to S3 directly."
    )

    output_name: str = Field(
        ...,
        title="Input N5 dataset path. Interpreted as relative to the bucket if given, otherwise,"
        "as a path on the local filesystem.",
    )


class ExaspimProcessingPipeline(AindModel):  # pragma: no cover
    """ExaSPIM processing pipeline configuration parameters"""
    n5_to_zarr: N5toZarrParameters = Field(..., title="N5 to multiscale Zarr conversion")


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


def print_example_manifest():  # pragma: no cover
    """Create example manifest file"""
    # print(ProcessingManifest.schema_json(indent=2))
    # print(ProcessingManifest.schema())

    processing_manifest_example = ExaspimManifest(
        specimen_id="000000",
        institution=Institution.AIND,
        processing_pipeline=ExaspimProcessingPipeline(
            n5_to_zarr=N5toZarrParameters(
                voxel_size_zyx=(1.0, 0.75, 0.75),
                input_bucket="aind-scratch-data",
                input_name="/gabor.kovacs/2023-07-25_1653_BSS_fusion_653431/ch561/",
                output_bucket="aind-scratch-data",
                output_name="/gabor.kovacs/n5_to_zarr_CO_2023-08-17_1351/",
            )
        ),
    )

    print(processing_manifest_example.json(indent=3))


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


if __name__ == "__main__":
    print_example_manifest()

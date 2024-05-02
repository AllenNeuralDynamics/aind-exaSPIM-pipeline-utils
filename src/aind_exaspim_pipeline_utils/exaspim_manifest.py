"""Manifest declaration for the exaSPIM capsules"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional, Tuple, List, Union

from aind_data_schema import DataProcess
from aind_data_schema.base import AindModel
from pydantic import Field, validator
import argparse

from .imagej_macros import ImagejMacros


# Based on aind-data-transfer/scripts/processing_manifest.py


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


class IJWrapperParameters(AindModel):  # pragma: no cover
    """ImageJ wrapper memory and parallelization runtime parameters"""

    memgb: int = Field(
        ...,
        title="Allowed JVM heap memory in GB."
        " Should be about 0.8 GB x number of parallel threads less than total available.",
        ge=16,
    )
    parallel: int = Field(..., title="Number of parallel Java worker threads.", ge=1, lt=128)

    # The IJ wrapper capsule does not read from this location until s3 support is fixed
    # but uses in the bigstitcher_emr.xml output file as input location. Also
    # propagated to the processing.json metadata file.
    input_uri: Optional[str] = Field(
        None,
        title="Input Zarr group dataset path. This is the dataset the alignment is running on."
        "Must be the aind-open-data s3:// path",
    )

    # The IJ wrapper capsule uploads the CO results folder here as a data product.
    output_uri: Optional[str] = Field(
        None,
        title="The capsule-s output location for the alignment dataset in s3:// "
        "including the full name of the top-level output s3 folder",
    )


class IPDetectionParameters(AindModel):  # pragma: no cover
    """Interest point detection parameters"""

    # dataset_xml: str = Field(..., title="NOT USED. BigStitcher xml dataset file in s3.")
    IJwrap: IJWrapperParameters = Field(..., title="ImageJ wrapper settings")

    downsample: int = Field(4, title="Downsampling factor. Use the one that is available in the dataset.")
    bead_choice: str = Field("manual", title="Beads detection mode")
    sigma: float = Field(1.8, title="Difference of Gaussians sigma (beads_mode==manual only)")
    threshold: float = Field(
        0.1, title="Difference of Gaussians detection threshold " "(beads_mode==manual only)."
    )
    find_minima: bool = Field(False, title="Find minima (beads_mode==manual only).")
    find_maxima: bool = Field(True, title="Find maxima (beads_mode==manual only).")
    set_minimum_maximum: bool = Field(False, title="Define the minimum and maximum intensity range manually")
    minimal_intensity: int = Field(0, title="Minimal intensity value (if set_minimum_maximum==True).")
    maximal_intensity: int = Field(65535, title="Minimal intensity value (if set_minimum_maximum==True).")
    maximum_number_of_detections: int = Field(
        0, title="If not equal to 0, the number of maximum IPs to detect. Set ip_limitation_choice, too."
    )
    ip_limitation_choice: str = Field(
        ..., title="How to pick limit_amount_of_detections is set >0 and the maximum number is hit."
    )

    @validator("bead_choice")
    def validate_bead_choice(cls, v: str) -> str:
        """Validate bead choice."""
        if v in ImagejMacros.MAP_BEAD_CHOICE.keys():
            return v
        else:
            raise ValueError(
                "bead_choice must be one of {}".format(list(ImagejMacros.MAP_BEAD_CHOICE.keys()))
            )

    @validator("ip_limitation_choice")
    def validate_ip_limitation_choice(cls, v: str) -> str:
        """Validate ip limitation choice."""
        if v in ImagejMacros.MAP_IP_LIMITATION_CHOICE.keys():
            return v
        else:
            raise ValueError(
                "ip_limitation_choice must be one of {}".format(
                    list(ImagejMacros.MAP_IP_LIMITATION_CHOICE.keys())
                )
            )


class IPRegistrationParameters(AindModel):  # pragma: no cover
    """Interest point based registration parameters"""

    # dataset_xml: str = Field(..., title="BigStitcher xml dataset file.")
    IJwrap: IJWrapperParameters = Field(..., title="ImageJ wrapper settings")
    transformation_choice: str = Field(..., title="Translation, rigid or full affine transformation ?")
    compare_views_choice: str = Field(..., title="Which views to compare ?")
    interest_point_inclusion_choice: str = Field(..., title="Which interest points to use ?")
    fix_views_choice: str = Field(..., title="Which views to fix ?")
    fixed_tile_ids: List[int] = Field(
        [
            0,
        ],
        title="Setup ids of fixed tiles (fix_views_choice==select_fixed).",
    )
    map_back_views_choice: str = Field(..., title="How to map back views?")
    map_back_reference_view: int = Field(0, title="Selected reference view for map back.")
    do_regularize: bool = Field(False, title="Do regularize transformation?")
    regularization_lambda: float = Field(0.1, title="Regularization lambda (do_regularize==True only).")
    regularize_with_choice: str = Field(
        "rigid", title="Which regularization to use (do_regularize==True only) ?"
    )

    @validator("transformation_choice")
    def validate_transformation_choice(cls, v: str) -> str:
        """Validate transformation choice"""
        if v in ImagejMacros.MAP_TRANSFORMATION.keys():
            return v
        else:
            raise ValueError(
                "transformation_choice must be one of {}".format(list(ImagejMacros.MAP_TRANSFORMATION.keys()))
            )

    @validator("compare_views_choice")
    def validate_compare_views_choice(cls, v: str) -> str:
        """Validate compare views choice."""
        if v in ImagejMacros.MAP_COMPARE_VIEWS.keys():
            return v
        else:
            raise ValueError(
                "compare_views_choice must be one of {}".format(list(ImagejMacros.MAP_COMPARE_VIEWS.keys()))
            )

    @validator("interest_point_inclusion_choice")
    def validate_interest_point_inclusion_choice(cls, v: str) -> str:
        """Validate interest point inclusion choice"""
        if v in ImagejMacros.MAP_INTEREST_POINT_INCLUSION.keys():
            return v
        else:
            raise ValueError(
                "interest_point_inclusion_choice must be one of {}".format(
                    list(ImagejMacros.MAP_INTEREST_POINT_INCLUSION.keys())
                )
            )

    @validator("fix_views_choice")
    def validate_fix_views_choice(cls, v: str) -> str:
        """Validate fix views choice"""
        if v in ImagejMacros.MAP_FIX_VIEWS.keys():
            return v
        else:
            raise ValueError(
                "fix_views_choice must be one of {}".format(list(ImagejMacros.MAP_FIX_VIEWS.keys()))
            )

    @validator("map_back_views_choice")
    def validate_map_back_views_choice(cls, v: str) -> str:
        """Validate map back views choice"""
        if v in ImagejMacros.MAP_MAP_BACK_VIEWS.keys():
            return v
        else:
            raise ValueError(
                "map_back_views_choice must be one of {}".format(list(ImagejMacros.MAP_MAP_BACK_VIEWS.keys()))
            )

    @validator("regularize_with_choice")
    def validate_regularize_with_choice(cls, v: str) -> str:
        """Validate regularize with choice"""
        if v in ImagejMacros.MAP_REGULARIZATION.keys():
            return v
        else:
            raise ValueError(
                "regularize_with_choice must be one of {}".format(
                    list(ImagejMacros.MAP_REGULARIZATION.keys())
                )
            )


class SparkInterestPointDetections(AindModel):  # pragma: no cover
    """Interest point detection parameters"""

    label: str = Field("beads", title="Label of the interest points")
    sigma: float = Field(4.0, title="sigma for segmentation, e.g. 1.8")
    threshold: float = Field(0.0015, title="threshold for segmentation, e.g. 0.008")
    overlappingOnly: bool = Field(True, title="Find overlapping interest points only")
    storeIntensities: bool = Field(True, title="Store intensities")
    prefetch: bool = Field(True, title="Prefetch")
    minIntensity: int = Field(0, title="Minimal intensity value")
    maxIntensity: int = Field(2000, title="Maximal intensity value")
    dsxy: int = Field(4, title="Downsampling factor for x and y")
    dsz: int = Field(4, title="Downsampling factor for z")
    blockSizeString: str = Field("1024,1024,1024", title="Block size string")
    type: str = Field("MAX", title="the type of interestpoints to find, MIN, MAX or BOTH (default: MAX)")
    localization: str = Field(
        "QUADRATIC", title="Subpixel localization method, NONE or QUADRATIC (default: QUADRATIC)"
    )


class SparkGeometricDescriptorMatching(AindModel):  # pragma: no cover
    """Geometric descriptor matching parameters"""

    label: str = Field("beads", title="Label")
    registrationMethod: str = Field(
        "PRECISE_TRANSLATION",
        title="the matching method; FAST_ROTATION, FAST_TRANSLATION, PRECISE_TRANSLATION or ICP",
    )
    significance: float = Field(
        3.0,
        title="how much better the first match between two descriptors has "
        "to be compareed to the second best one (default: 3.0)",
    )
    redundancy: int = Field(1, title="the redundancy of the local descriptor (default: 1)")
    numNeighbors: int = Field(
        3,
        title="the number of neighboring points used to build the local descriptor,"
        " only supported by PRECISE_TRANSLATION (default: 3)",
    )
    clearCorrespondences: bool = Field(False, title="Clear Correspondences")
    interestpointsForReg: str = Field(
        "OVERLAPPING_ONLY",
        title="which interest points to use for pairwise registrations, "
        "use OVERLAPPING_ONLY or ALL points (default: ALL)",
    )
    viewReg: str = Field(
        "OVERLAPPING_ONLY",
        title="which views to register with each other, compare OVERLAPPING_ONLY "
        "or ALL_AGAINST_ALL (default: OVERLAPPING_ONLY)",
    )
    interestPointMergeDistance: float = Field(5.0, title="Interest Point Merge Distance")
    groupIllums: bool = Field(False, title="Group Illuminations")
    groupChannels: bool = Field(False, title="Group Channels")
    groupTiles: bool = Field(False, title="Group Tiles")
    splitTimepoints: bool = Field(False, title="Split Timepoints")
    ransacIterations: Optional[int] = Field(
        None, title="max number of ransac iterations (default: 10,000 for descriptors, 200 for ICP)"
    )
    ransacMaxError: Optional[float] = Field(
        None, title="ransac max error in pixels (default: 5.0 for descriptors, 2.5 for ICP)"
    )
    ransacMinInlierRatio: float = Field(0.1, title="RANSAC Minimum Inlier Ratio")
    ransacMinInlierFactor: float = Field(
        3.0,
        title="ransac min inlier factor, i.e. how many time the minimal number of matches need to found, "
        "e.g. affine needs 4 matches, 3x means at least 12 matches required (default: 3.0)",
    )
    icpMaxError: float = Field(5.0, title="ICP max error in pixels (default: 5.0)")
    icpIterations: int = Field(200, title="max number of ICP iterations (default: 200)")
    icpUseRANSAC: bool = Field(False, title="ICP Use RANSAC")

    registrationTP: str = Field(
        "TIMEPOINTS_INDIVIDUALLY",
        title="time series registration type; TIMEPOINTS_INDIVIDUALLY (i.e. no registration across time), "
        "TO_REFERENCE_TIMEPOINT, ALL_TO_ALL or ALL_TO_ALL_WITH_RANGE "
        "(default: TIMEPOINTS_INDIVIDUALLY)",
    )
    referenceTP: Optional[int] = Field(
        None, title="the reference timepoint if timepointAlign == REFERENCE (default: first timepoint)"
    )
    rangeTP: int = Field(
        5, title="the range of timepoints if timepointAlign == ALL_TO_ALL_RANGE (default: 5)"
    )
    transformationModel: str = Field(
        "AFFINE", title="which transformation model to use; TRANSLATION, RIGID or AFFINE (default: AFFINE)"
    )
    regularizationModel: str = Field(
        "RIGID",
        title="which regularization model to use; NONE, IDENTITY, "
        "TRANSLATION, RIGID or AFFINE (default: RIGID)",
    )
    regularizationLambda: float = Field(0.1, title="lamdba to use for regularization model (default: 0.1)")


class Solver(AindModel):  # pragma: no cover
    """Solver parameters"""

    sourcePoints: str = Field(
        "IP", title="which source to use for the solve, IP (interest points) or STITCHING"
    )
    groupIllums: Optional[bool] = Field(
        None,
        title="group all illumination directions that belong to the same angle/channel/tile/timepoint "
        "together as one view, e.g. to stitch illums as one "
        "(default: false for IP, true for stitching)",
    )
    groupChannels: Optional[bool] = Field(
        None,
        title="group all channels that belong to the same angle/illumination/tile/timepoint together "
        "as one view, e.g. to stitch channels as one (default: false for IP, true for stitching)",
    )
    groupTiles: Optional[bool] = Field(
        None,
        title="group all tiles that belong to the same angle/channel/illumination/timepoint together "
        "as one view, e.g. to align across angles (default: false)",
    )
    splitTimepoints: Optional[bool] = Field(
        None,
        title="group all angles/channels/illums/tiles that belong to the same timepoint as one View, "
        "e.g. for stabilization across time (default: false)",
    )
    label: Optional[str] = Field(
        "beads", title="label of the interest points used for solve if using interest points (e.g. beads)"
    )
    globalOptType: str = Field(
        "TWO_ROUND_SIMPLE",
        title="global optimization method; ONE_ROUND_SIMPLE, ONE_ROUND_ITERATIVE, TWO_ROUND_SIMPLE or "
        "TWO_ROUND_ITERATIVE. Two round handles unconnected tiles, iterative handles wrong links "
        "(default: ONE_ROUND_SIMPLE)",
    )
    relativeThreshold: float = Field(
        3.5,
        title="relative error threshold for iterative solvers, how many times worse than the average error "
        "a link needs to be (default: 3.5)",
    )
    absoluteThreshold: float = Field(
        7.0, title="absoluted error threshold for iterative solver to drop a link in pixels (default: 7.0)"
    )
    maxError: float = Field(5.0, title="max error for the solve (default: 5.0)")
    maxIterations: int = Field(10000, title="max number of iterations for solve (default: 10,000)")
    maxPlateauwidth: int = Field(200, title="max plateau witdth for solve (default: 200)")
    disableFixedViews: bool = Field(False, title="disable fixing of views (see --fixedViews)")
    fixedViews: Optional[List[str]] = Field(
        ["0,7"],
        title="define a list of (or a single) fixed view ids (time point, view setup), e.g. -fv '0,0' "
        "-fv '0,1' (default: first view id)",
    )

    registrationTP: str = Field(
        "TIMEPOINTS_INDIVIDUALLY",
        title="time series registration type; TIMEPOINTS_INDIVIDUALLY (i.e. no registration across time), "
        "TO_REFERENCE_TIMEPOINT, ALL_TO_ALL or ALL_TO_ALL_WITH_RANGE "
        "(default: TIMEPOINTS_INDIVIDUALLY)",
    )
    referenceTP: Optional[int] = Field(
        None, title="the reference timepoint if timepointAlign == REFERENCE (default: first timepoint)"
    )
    rangeTP: int = Field(
        5, title="the range of timepoints if timepointAlign == ALL_TO_ALL_RANGE (default: 5)"
    )
    transformationModel: str = Field(
        "AFFINE", title="which transformation model to use; TRANSLATION, RIGID or AFFINE (default: AFFINE)"
    )
    regularizationModel: str = Field(
        "RIGID",
        title="which regularization model to use; NONE, IDENTITY, "
        "TRANSLATION, RIGID or AFFINE (default: RIGID)",
    )
    regularizationLambda: float = Field(0.1, title="lamdba to use for regularization model (default: 0.1)")


class XMLCreationParameters(AindModel):  # pragma: no cover
    """XML converter capsule parameters."""

    ch_name: str = Field(..., title="Channel name, without the ch prefix")

    input_uri: Optional[str] = Field(
        None,
        title="Input Zarr group dataset path. This is the dataset the alignment is running on."
        "Must be the aind-open-data s3:// path without the SPIM.ome.zarr suffix",
    )


class ExaspimProcessingPipeline(AindModel):  # pragma: no cover
    """ExaSPIM processing pipeline configuration parameters

    If a field is None, it is considered to be a disabled step."""

    schema_version: str = Field("0.11.0", title="Schema Version", const=True)
    license: str = Field("CC-BY-4.0", title="License", const=True)

    creation_time: datetime = Field(
        ...,
        title="UTC Creation time of manifest",
    )
    pipeline_suffix: str = Field(..., title="Filename timestamp suffix")
    subject_id: str = Field(
        None, title="The subject id in accordance with the folder names and metadata files"
    )

    # In accordance with data_description. Which one is this for flat-fielded data?
    name: Optional[str] = Field(
        None,
        description="Name of data, conventionally also the name of "
        "the directory containing all data and metadata",
        title="Name",
    )
    spark_ip_detections: Union[
        None, SparkInterestPointDetections, List[SparkInterestPointDetections]
    ] = Field(None, title="Spark interest point detection")
    spark_geometric_descriptor_matching_tr: Union[None, SparkGeometricDescriptorMatching] = Field(
        None, title="Spark geometric descriptor matching"
    )
    solver_tr: Union[None, Solver] = Field(None, title="Solver parameters")
    spark_geometric_descriptor_matching_aff: Union[None, SparkGeometricDescriptorMatching] = Field(
        None, title="Spark geometric descriptor matching"
    )
    solver_aff: Union[None, Solver] = Field(None, title="Solver parameters")

    xml_creation: XMLCreationParameters = Field(None, title="XML creation")
    ip_detection: IPDetectionParameters = Field(None, title="Interest point detection")
    ip_registrations: List[IPRegistrationParameters] = Field(
        None, title="List of interest point registration steps."
    )
    n5_to_zarr: N5toZarrParameters = Field(None, title="N5 to single scale Zarr conversion")
    zarr_multiscale: ZarrMultiscaleParameters = Field(None, title="Zarr to multiscale Zarr conversion")


def create_example_manifest(printit=True) -> ExaspimProcessingPipeline | None:  # pragma: no cover
    """Create example manifest file

    Parameters
    ----------
    printit: bool
      Print the example?

    Returns
    -------
    example_manifest: ExaspimManifest
    """
    t = datetime.now()
    processing_manifest_example = ExaspimProcessingPipeline(
        creation_time=t,
        pipeline_suffix=t.strftime("%Y-%m-%d_%H-%M-%S"),
        subject_id="000000",
        n5_to_zarr=N5toZarrParameters(
            voxel_size_zyx=(1.0, 0.748, 0.748),
            input_uri="s3://aind-scratch-data/gabor.kovacs/2023-07-25_1653_BSS_fusion_653431/ch561/",
            output_uri="s3://aind-scratch-data/gabor.kovacs/n5_to_zarr_CO_2023-08-17_1351/",
        ),
        zarr_multiscale=ZarrMultiscaleParameters(
            voxel_size_zyx=(1.0, 0.748, 0.748),
            input_uri="s3://aind-scratch-data/gabor.kovacs/2023-07-25_1653_BSS_fusion_653431/ch561/",
        ),
    )

    if printit:
        print(processing_manifest_example.json(indent=3))
        return  # If printed, we assume call from the cli
    return processing_manifest_example


def get_capsule_manifest(
    args: Optional[argparse.Namespace] = None,
) -> ExaspimProcessingPipeline:  # pragma: no cover
    """Get the manifest file from its Code Ocean location or as given in the cmd-line argument.

    Raises
    ------
    If the manifest is not found, required fields will be missing at schema validation.
    """
    if args is not None:
        manifest_name = args.manifest_name
    else:
        manifest_name = "../data/manifest/exaspim_manifest.json"
    with open(manifest_name, "r") as f:
        json_data = json.load(f)
    return ExaspimProcessingPipeline(**json_data)


def get_capsule_metadata() -> dict:  # pragma: no cover
    """Get the metadata file from its Code Ocean location"""
    metadata_name = "../data/meta/metadata.json"
    if os.path.exists(metadata_name):
        with open(metadata_name) as json_file:
            json_data = json.load(json_file)
    else:
        json_data = {}
    # return Metadata.parse_obj(json_data)
    return json_data


# TODO: We do not yet use an accumulative metadata file with multiple data_process entries.
# def append_process_entries_to_metadata(
#     dataset_metadata: Metadata, processes: Iterable[DataProcess]
# ) -> None:  # pragma: no cover
#     """Append the given process metadata entries to the dataset_metadata
#
#     So long the pipeline is a linear sequence of steps, this should always be the
#     case. Otherwise the process metadata should be collected and appended
#     at the end of the parallel processing capsules.
#     """
#     for process in processes:
#         dataset_metadata.processing.data_processes.append(process)


def write_result_dataset_metadata(dataset_metadata: dict) -> None:  # pragma: no cover
    """Write the updated metadata file to the Code Ocean results folder."""
    os.makedirs("../results/meta", exist_ok=True)
    with open("../results/meta/metadata.json", "w") as f:
        json.dump(dataset_metadata, f, indent=3)


def write_process_metadata(capsule_metadata: DataProcess, prefix=None) -> None:  # pragma: no cover
    """Write the process.json file about this processing step to the Code Ocean results folder."""
    # os.makedirs("../results/meta", exist_ok=True)
    if prefix is None:
        prefix = ""
    else:
        prefix = prefix + "_"
    # with open(f"../results/meta/exaspim_{prefix}process.json", "w") as f:
    with open("../results/process_output.json", "w") as f:
        f.write(capsule_metadata.json(indent=3))


if __name__ == "__main__":  # pragma: no cover
    create_example_manifest(printit=True)

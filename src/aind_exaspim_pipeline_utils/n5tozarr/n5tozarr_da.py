"""ExaSPIM cloud conversion of N5 to multiscale ZARR using Dask.array"""
import datetime
import logging
import sys

import xarray_multiscale
from aind_data_schema import DataProcess
from aind_data_schema.processing import ProcessName
from aind_data_transfer.transformations.ome_zarr import _get_first_mipmap_level
from aind_data_transfer.util.io_utils import BlockedArrayWriter
from numcodecs.abc import Codec

import aind_exaspim_pipeline_utils
from xarray_multiscale.reducers import WindowedReducer

import multiprocessing
from typing import Tuple
import time
import psutil
import zarr
import re
from numcodecs import Blosc
import dask
import dask.array
from dask.distributed import Client
from aind_data_transfer.transformations import ome_zarr
from aind_data_transfer.util import chunk_utils, io_utils
from aind_exaspim_pipeline_utils import exaspim_manifest
from aind_exaspim_pipeline_utils.exaspim_manifest import write_process_metadata


def fmt_uri(uri: str) -> str:
    """Format location paths by making slash usage consistent.

    All multiple occurrence internal slashes are replaced to single ones.

    Local paths can be relative or absolute, trailing slash will be added if missing.

    S3 references formatted to pattern ``s3://bucket/path/``.

    Parameters
    ----------
    uri: `str`
        The resource URI.

    Returns
    -------
    r: `str`
      Formatted uri, either a local file system path or an s3:// uri.
    """
    if uri.startswith("s3:") or uri.startswith("S3:"):
        s = "/".join(("", *uri[3:].split("/"), ""))
        return "s3:/" + re.sub(r"/{2,}", "/", s)
    else:
        s = "/".join((*uri.split("/"), ""))
        return re.sub(r"/{2,}", "/", s)


def downsample_and_store(
    arr: dask.array.Array,
    group: zarr.Group,
    n_lvls: int,
    scale_factors: Tuple,
    block_shape: Tuple,
    compressor: Codec = None,
    reducer: WindowedReducer = xarray_multiscale.reducers.windowed_mean,
    fromLevel: int = 1,
) -> list:  # pragma: no cover
    """
    Progressively downsample the input array and store the results as separate arrays in a Zarr group.

    Parameters
    ----------
    arr : da.Array
        The full-resolution Dask array.
    group : zarr.Group
        The output Zarr group.
    n_lvls : int
        The number of pyramid levels.
    scale_factors : Tuple
        The scale factors for downsampling along each dimension.
    block_shape : Tuple
        The shape of blocks to use for partitioning the array.
    compressor : numcodecs.abc.Codec, optional
        The compression codec to use for the output Zarr array. Default is Blosc with "zstd"
        method and compression level 1.
    fromLevel : int
        The first downscaled level to write. `arr` must represent fromLevel - 1. Defaults to 1.
    """

    for arr_index in range(fromLevel, n_lvls):
        LOGGER.info("Creating downsampled level %d in dask.", arr_index)
        first_mipmap = _get_first_mipmap_level(arr, scale_factors, reducer)

        LOGGER.info("Creating dataset for level %d.", arr_index)
        ds = group.create_dataset(
            str(arr_index),
            shape=first_mipmap.shape,
            chunks=first_mipmap.chunksize,
            dtype=first_mipmap.dtype,
            compressor=compressor,
            dimension_separator="/",
            overwrite=True,
        )
        LOGGER.info("Storing downsampled level %d.", arr_index)
        BlockedArrayWriter.store(first_mipmap, ds, block_shape)

        arr = dask.array.from_array(ds, chunks=ds.chunks)


def run_n5tozarr(
    input_uri: str,
    output_uri: str,
    voxel_sizes_zyx: Tuple[float, float, float],
):  # pragma: no cover
    """Run initial conversion and 4 layers of downscaling.

    All output arrays will be 5D, chunked as (1, 1, 128, 128, 128),
    downscaling factor is (1, 1, 2, 2, 2).

    Parameters
    ----------
    input_uri: `str`
        Input uri or path on the local filesystem.
    output_uri: `str`
        Output uri or on the local filesystem.
    voxel_sizes_zyx: tuple of `float`
        Voxel size in microns in the input (full resolution) dataset
    """
    LOGGER.debug("Initialize source N5 store")
    n5s = zarr.n5.N5FSStore(input_uri)
    zg = zarr.open(store=n5s, mode="r")
    LOGGER.debug("Initialize dask array from N5 source")
    arr = dask.array.from_array(zg["s0"], chunks=zg["s0"].chunks)
    arr = chunk_utils.ensure_array_5d(arr)
    LOGGER.debug("Re-chunk dask array to desired output chunk size.")
    arr = arr.rechunk((1, 1, 256, 256, 256))

    LOGGER.info(f"Input array: {arr}")
    LOGGER.info(f"Input array size: {arr.nbytes / 2 ** 20} MiB")

    LOGGER.debug("Initialize target Zarr store")
    output_path = output_uri
    group = zarr.open_group(output_path, mode="a")

    scale_factors = (2, 2, 2)
    scale_factors = chunk_utils.ensure_shape_5d(scale_factors)

    n_levels = 1
    compressor = Blosc(cname="zstd", clevel=1)

    block_shape = chunk_utils.ensure_shape_5d(
        io_utils.BlockedArrayWriter.get_block_shape(arr, target_size_mb=1048576)  # 8k**3 blocks of 16bit data
    )
    LOGGER.info(f"Calculation block shape: {block_shape}")

    # Actual Processing
    ome_zarr.write_ome_ngff_metadata(
        group,
        arr,
        output_uri,
        n_levels,
        scale_factors[2:],
        voxel_sizes_zyx,
        origin=None,
    )

    t0 = time.time()
    LOGGER.info("Starting initial N5 -> Zarr copy.")
    ome_zarr.store_array(arr, group, "0", block_shape, compressor)
    write_time = time.time() - t0
    LOGGER.info(f"Finished writing tile. Took {write_time}s.")


def run_zarr_multiscale(
    input_uri: str, output_uri: str, voxel_sizes_zyx: Tuple[float, float, float], fromLevel: int = 1
):  # pragma: no cover
    """Run downscaling on an existing 0 level zarr.

    All output arrays will be 5D, chunked as (1, 1, 128, 128, 128),
    downscaling factor is (1, 1, 2, 2, 2).

    Parameters
    ----------
    input_uri: `str`
        Input s3 uri path or path on the local filesystem.
    output_uri: `str`
        Output s3 uri path or path on the local filesystem. Can be the same as `input_uri`.
    voxel_sizes_zyx: tuple of `float`
        Voxel size in microns in the input (full resolution) dataset
    """
    LOGGER.debug("Initialize source Zarr store")
    zg = zarr.open_group(input_uri, mode="r")
    LOGGER.info("Get dask array from Zarr source for full resolution")
    arrZero = dask.array.from_array(
        zg["0"], chunks=zg["0"].chunks
    )  # For metadata writing we need the full resolution shape
    arrZero = chunk_utils.ensure_array_5d(arrZero)
    arrZero = arrZero.rechunk((1, 1, 256, 256, 256))

    LOGGER.info(f"Full resolution array: {arrZero}")
    LOGGER.info(f"Full resolution input array size: {arrZero.nbytes / 2 ** 20} MiB")

    LOGGER.debug("Initialize target Zarr store")
    group = zarr.open_group(output_uri, mode="a")

    scale_factors = (2, 2, 2)
    scale_factors = chunk_utils.ensure_shape_5d(scale_factors)

    n_levels = 8
    compressor = Blosc(cname="zstd", clevel=1)

    # Actual Processing
    ome_zarr.write_ome_ngff_metadata(
        group,
        arrZero,
        output_uri,
        n_levels,
        scale_factors[2:],
        voxel_sizes_zyx,
        origin=None,
    )

    if fromLevel > 1:
        prevLevel = str(fromLevel - 1)
        LOGGER.info("Initialize dask source array from Zarr source level %s", prevLevel)
        arr = dask.array.from_array(zg[prevLevel], chunks=zg[prevLevel].chunks)
        arr = chunk_utils.ensure_array_5d(arr)
        arr = arr.rechunk((1, 1, 256, 256, 256))
    else:
        arr = arrZero

    del arrZero  # Can be garbage collected if different from arr

    block_shape = chunk_utils.ensure_shape_5d(
        io_utils.BlockedArrayWriter.get_block_shape(arr, target_size_mb=1048576)  # 8k**3 blocks of 16bit data
    )
    LOGGER.info(f"Calculation block shape: {block_shape}")

    t0 = time.time()
    LOGGER.info("Starting N5 -> downsampled Zarr level copies.")
    downsample_and_store(arr, group, n_levels, scale_factors, block_shape, compressor, fromLevel=fromLevel)
    write_time = time.time() - t0

    LOGGER.info("Finished writing tile. Took %d s.", write_time)


def get_worker_memory(n_worker):  # pragma: no cover
    """Determine the per-worker memory"""
    total = psutil.virtual_memory().total
    GByte = 1024 * 1024 * 1024
    LOGGER.info("Total physical memory: %.1f GiB", total / GByte)
    wmem = total - 24 * GByte  # Reserve for scheduler
    perworker = wmem // n_worker
    if wmem < 0 or perworker < 2 * GByte:
        raise RuntimeError("Not enough memory for 24 GiB for scheduler and at least 2 GiB per worker")
    LOGGER.info("Set aside 24 GiB for scheduler, %.1f GiB per worker process", perworker / GByte)
    return perworker


def set_worker_logging(level: int = logging.DEBUG):  # pragma: no cover
    """Configure logger levels on a worker."""
    logging.getLogger().setLevel(level)
    logging.getLogger("distributed").setLevel(level)
    # Do not want to see network debug stuff
    if level > logging.INFO:
        infolevel = level
    else:
        infolevel = logging.INFO
    logging.getLogger("boto3").setLevel(infolevel)
    logging.getLogger("botocore").setLevel(infolevel)
    logging.getLogger("s3fs").setLevel(infolevel)
    logging.getLogger("urllib3").setLevel(infolevel)


def config_client_logging(level: int = logging.DEBUG):  # pragma: no cover
    """Configure logging on a worker or the client."""
    # config = {
    #     "version": 1,
    #     "handlers": {
    #         "console": {
    #             "class": "logging.StreamHandler",
    #             "formatter": "default",
    #             "level": logging.getLevelName(level),
    #         }
    #     },
    #     "formatters": {
    #         "default": {
    #             # "worker" field was referenced in `Client.forward_logging` documentation but does not work:
    #             # "format": "%(asctime)s %(levelname)-8s [%(process)d %(worker)s] %(name)-15s %(message)s",
    #             "format": "%(asctime)s [%(process)d] %(levelname)s %(name)s %(message)s",
    #             "datefmt": "%Y-%m-%d %H:%M:%S",
    #         }
    #     },
    #     "root": {"handlers": ["console"]},
    # }
    R = logging.getLogger()
    for hdlr in R.handlers[:]:  # remove all old handlers
        R.removeHandler(hdlr)

    h1 = logging.StreamHandler(sys.stderr)
    h1.setFormatter(logging.Formatter("%(asctime)s [%(process)d] %(levelname)s %(name)s %(message)s"))
    logging.getLogger().setLevel(level)
    logging.getLogger().addHandler(h1)
    # logging.config.dictConfig(config)
    # logging.getLogger().setLevel(level)
    # Do not want to see network debug stuff
    if level > logging.INFO:
        infolevel = level
    else:
        infolevel = logging.INFO
    logging.getLogger("distributed.worker").setLevel(infolevel)
    logging.getLogger("distributed").setLevel(infolevel)
    logging.getLogger("boto3").setLevel(infolevel)
    logging.getLogger("botocore").setLevel(infolevel)
    logging.getLogger("s3fs").setLevel(infolevel)
    logging.getLogger("urllib3").setLevel(infolevel)


def n5tozarr_da_converter():  # pragma: no cover
    """Main entry point to n5tozarr converter task."""
    config_client_logging(logging.INFO)
    global LOGGER
    LOGGER = logging.getLogger("n5tozarr")
    LOGGER.setLevel(logging.INFO)

    capsule_manifest = exaspim_manifest.get_capsule_manifest()
    LOGGER.info("This is pipeline run: %s", capsule_manifest.pipeline_suffix)
    config = capsule_manifest.n5_to_zarr.dict()
    if config is None:
        raise ValueError("Manifest does not contain configuration for n5tozarr processing")

    config["input_uri"] = fmt_uri(config["input_uri"])
    config["output_uri"] = fmt_uri(config["output_uri"])
    n_cpu = multiprocessing.cpu_count()

    config["capsule"] = dict(version="n5tozarr_0.1.0", n_cpu=n_cpu)  # TBD: obtain version from CO environment

    # Create initial metadata in case the run crashes
    process_meta = get_n5tozarr_metadata(config)
    write_process_metadata(process_meta, prefix="n5tozarr")

    # Start dask cluster
    LOGGER.info("Starting local Dask cluster with %d processes and 2 threads per process.", n_cpu)
    dask.config.set(
        {
            "distributed.worker.memory.spill": False,  # Do not spill to /tmp space in a capsule
            "distributed.worker.memory.target": False,  # Do not spill to /tmp space in a capsule
            # If a worker gets paused, after the recent-to-old wait time, we hope
            # that it will cross the termination limit and eventually gets restarted
            "distributed.worker.memory.terminate": 0.95,
            # Pause and wait for task finish then GC and trimming:
            "distributed.worker.memory.pause": 0.94,
            # Do not receive data from other workers:
            "distributed.worker.memory.rebalance.recipient-max": 0.05,
            # Should be longer than typical task runtime:
            "distributed.worker.memory.recent-to-old-time": "300s",
        }
    )
    client = Client(
        n_workers=n_cpu,
        threads_per_worker=2,
        memory_limit=get_worker_memory(n_cpu),
        processes=True,
        silence_logs=logging.INFO,
    )
    client.run(set_worker_logging, logging.INFO)
    client.forward_logging()
    # Run jobs
    run_n5tozarr(config["input_uri"], config["output_uri"], config["voxel_size_zyx"])
    # Update metadata to show that we've finished properly
    set_metadata_done(process_meta)
    write_process_metadata(process_meta, prefix="n5tozarr")
    # append_metadata_to_manifest(capsule_manifest, meta)
    # write_result_manifest(capsule_manifest)
    # Close down
    LOGGER.info("Sleep 120s to get workers into an idle state.")
    time.sleep(120)  # leave time for workers to get into an idle state before shutting down
    LOGGER.info("Closing cluster.")
    client.close(180)  # leave time for workers to exit
    LOGGER.info("Done.")


def get_zarr_multiscale_metadata(config: dict):  # pragma: no cover
    """Initiate metadata instance with current timestamp and configuration."""
    t = datetime.datetime.now()
    dp = DataProcess(
        name=ProcessName.FILE_CONVERSION,
        software_version=config["capsule"]["version"],
        start_date_time=t,
        end_date_time=t,
        input_location=config["input_uri"],
        output_location=config["output_uri"],
        code_url="https://github.com/AllenNeuralDynamics/aind-exaSPIM-pipeline-utils",
        code_version=aind_exaspim_pipeline_utils.__version__,
        parameters=config,
        outputs=None,
        notes="IN PROGRESS",
    )
    return dp


def get_n5tozarr_metadata(config: dict):  # pragma: no cover
    """Initiate metadata instance with current timestamp and configuration."""
    t = datetime.datetime.now()
    dp = DataProcess(
        name=ProcessName.FILE_CONVERSION,
        software_version=config["capsule"]["version"],
        start_date_time=t,
        end_date_time=t,
        input_location=config["input_uri"],
        output_location=config["output_uri"],
        code_url="https://github.com/AllenNeuralDynamics/aind-exaSPIM-pipeline-utils",
        code_version=aind_exaspim_pipeline_utils.__version__,
        parameters=config,
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


def zarr_multiscale_converter():  # pragma: no cover
    """Main entry point for zarr downscaling task."""
    config_client_logging(logging.INFO)
    global LOGGER
    LOGGER = logging.getLogger("zarr_mscale")
    LOGGER.setLevel(logging.INFO)

    capsule_manifest = exaspim_manifest.get_capsule_manifest()
    LOGGER.info("This is pipeline run: %s", capsule_manifest.pipeline_suffix)
    config = capsule_manifest.zarr_multiscale.dict()
    if config is None:
        raise ValueError("Manifest does not contain configuration for zarr_multiscale processing")

    # Add dynamic entries to config
    config["input_uri"] = fmt_uri(config["input_uri"])
    if config["output_uri"] is None:
        config["output_uri"] = config["input_uri"]
    else:
        config["output_uri"] = fmt_uri(config["output_uri"])
    n_cpu = multiprocessing.cpu_count()
    config["capsule"] = dict(
        version="zarr_multiscale_0.1.0", n_cpu=n_cpu
    )  # TBD: obtain version from CO environment

    # Create initial metadata in case the run crashes
    process_meta = get_zarr_multiscale_metadata(config)
    write_process_metadata(process_meta, prefix="zarr_multiscale")

    # Start dask cluster
    LOGGER.info("Starting local Dask cluster with %d processes and 2 threads per process.", n_cpu)
    dask.config.set(
        {
            "distributed.worker.memory.spill": False,  # Do not spill to /tmp space in a capsule
            "distributed.worker.memory.target": False,  # Do not spill to /tmp space in a capsule
            # If a worker gets paused, after the recent-to-old wait time, we hope
            # that it will cross the termination limit and eventually gets restarted
            "distributed.worker.memory.terminate": 0.95,
            # Pause and wait for task finish then GC and trimming:
            "distributed.worker.memory.pause": 0.94,
            # Do not receive data from other workers:
            "distributed.worker.memory.rebalance.recipient-max": 0.05,
            # Should be longer than typical task runtime:
            "distributed.worker.memory.recent-to-old-time": "300s",
        }
    )
    client = Client(
        n_workers=n_cpu,
        threads_per_worker=2,
        memory_limit=get_worker_memory(n_cpu),
        processes=True,
        silence_logs=logging.INFO,
    )
    client.run(set_worker_logging, logging.INFO)
    client.forward_logging()
    # Run jobs
    run_zarr_multiscale(config["input_uri"], config["output_uri"], config["voxel_size_zyx"])
    # Update metadata to show that we've finished properly
    set_metadata_done(process_meta)
    write_process_metadata(process_meta, prefix="zarr_multiscale")
    # append_metadata_to_manifest(capsule_manifest, meta)
    # write_result_manifest(capsule_manifest)
    # Close down
    LOGGER.info("Sleep 120s to get workers into an idle state.")
    time.sleep(120)  # leave time for workers to get into an idle state before shutting down
    LOGGER.info("Closing cluster.")
    client.close(180)  # leave time for workers to exit
    LOGGER.info("Done.")


if __name__ == "__main__":
    n5tozarr_da_converter()

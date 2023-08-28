"""ExaSPIM cloud conversion of N5 to multiscale ZARR using Dask.array"""
import logging
import multiprocessing
from typing import Iterable, Optional, Tuple
import time
import zarr
import re
from numcodecs import Blosc

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(process)d %(message)s", datefmt="%Y-%m-%d %H:%M"
)
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

import dask  # noqa: E402
import dask.array  # noqa: E402
from dask.distributed import Client  # noqa: E402
from aind_data_transfer.transformations import ome_zarr  # noqa: E402
from aind_data_transfer.util import chunk_utils, io_utils  # noqa: E402
from aind_exaspim_pipeline_utils import exaspim_manifest  # noqa: E402
from aind_exaspim_pipeline_utils.exaspim_manifest import N5toZarrParameters  # noqa: E402


def get_uri(bucket_name: Optional[str], *names: Iterable[str]) -> str:
    """Format location paths by making slash usage consistent.

    All multiple occurrence internal slashes are replaced to single ones.

    Parameters
    ----------
    bucket_name: `str`, optional
        The name of the S3 bucket. If specified, it triggers
        the interpretation as an S3 uri.

    names: Iterable of `str`
        Path elements to connect with '/'.

    Returns
    -------
    r: `str`
      Formatted uri, either a local file system path or an s3:// uri.
    """
    if bucket_name is None:
        s = "/".join((*names, ""))
        return re.sub(r"/{2,}", "/", s)
    s = "/".join(("", bucket_name, *names, ""))
    return "s3:/" + re.sub(r"/{2,}", "/", s)


def run_multiscale(
        input_bucket: Optional[str],
        input_name: str,
        output_bucket: Optional[str],
        output_name: str,
        voxel_sizes_zyx: Tuple[float, float, float],
):  # pragma: no cover
    """Run initial conversion and 4 layers of downscaling.

    All output arrays will be 5D, chunked as (1, 1, 128, 128, 128),
    downscaling factor is (1, 1, 2, 2, 2).

    Parameters
    ----------
    input_bucket: `str`, optional
        Input bucket or None for local filesystem access.
    input_name: `str`
        Input path within bucket or on the local filesystem.
    output_bucket: `str`, optional
        Output bucket or None for local filesystem access.
    output_name: `str`
        Output path within bucket or on the local filesystem.
    voxel_sizes_zyx: tuple of `float`
        Voxel size in microns in the input (full resolution) dataset
    """
    LOGGER.debug("Initialize source N5 store")
    n5s = zarr.n5.N5FSStore(get_uri(input_bucket, input_name))
    zg = zarr.open(store=n5s, mode="r")
    LOGGER.debug("Initialize dask array from N5 source")
    arr = dask.array.from_array(zg["s0"])
    arr = chunk_utils.ensure_array_5d(arr)
    LOGGER.debug("Re-chunk dask array to desired output chunk size.")
    arr = arr.rechunk((1, 1, 128, 128, 128))

    LOGGER.info(f"Input array: {arr}")
    LOGGER.info(f"Input array size: {arr.nbytes / 2 ** 20} MiB")

    LOGGER.debug("Initialize target Zarr store")
    output_path = get_uri(output_bucket, output_name)
    group = zarr.open_group(output_path, mode="w")

    scale_factors = (2, 2, 2)
    scale_factors = chunk_utils.ensure_shape_5d(scale_factors)

    n_levels = 5
    compressor = Blosc(cname="zstd", clevel=1)

    block_shape = chunk_utils.ensure_shape_5d(
        io_utils.BlockedArrayWriter.get_block_shape(arr, target_size_mb=819200)
    )
    LOGGER.info(f"Calculation block shape: {block_shape}")

    # Actual Processing
    ome_zarr.write_ome_ngff_metadata(
        group,
        arr,
        output_name,
        n_levels,
        scale_factors[2:],
        voxel_sizes_zyx,
        origin=None,
    )

    t0 = time.time()
    LOGGER.info("Starting initial N5 -> Zarr copy.")
    ome_zarr.store_array(arr, group, "0", block_shape, compressor)
    LOGGER.info("Starting N5 -> downsampled Zarr level copies.")
    pyramid = ome_zarr.downsample_and_store(arr, group, n_levels, scale_factors, block_shape, compressor)
    write_time = time.time() - t0

    LOGGER.info(
        f"Finished writing tile.\n"
        f"Took {write_time}s. {ome_zarr._get_bytes(pyramid) / write_time / (1024 ** 2)} MiB/s"
    )


def n5tozarr_da_converter():  # pragma: no cover
    """Main entry point."""

    config: N5toZarrParameters = exaspim_manifest.get_capsule_manifest().processing_pipeline.n5_to_zarr
    n_cpu = multiprocessing.cpu_count()
    LOGGER.info("Starting local Dask cluster with %d processes and 2 threads per process.", n_cpu)
    dask.config.set(
        {
            "distributed.worker.memory.spill": False,  # Do not spill to /tmp space in a capsule
            "distributed.worker.memory.target": False,  # Do not spill to /tmp space in a capsule
            "distributed.worker.memory.terminate": False,  # Just pause and wait for GC and memory trimming
        }
    )
    client = Client(n_workers=n_cpu, threads_per_worker=2)
    run_multiscale(
        config.input_bucket,
        config.input_name,
        config.output_bucket,
        config.output_name,
        config.voxel_size_zyx,
    )
    client.close(180)  # leave time for workers to exit


if __name__ == "__main__":
    n5tozarr_da_converter()

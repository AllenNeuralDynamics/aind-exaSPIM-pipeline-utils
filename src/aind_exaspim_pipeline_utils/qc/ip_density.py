"""Interestpoint density quality control module."""

from __future__ import annotations

import logging
import sys
from typing import Optional

import numpy as np

import xmltodict
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from .tile_transformations import (
    get_tile_overlapping_IPs,
    get_tile_pair_overlap,
    filter_tile_corresponding_IPs,
    format_large_numbers,
    read_tiles_interestpoints,
    read_ip_correspondences,
    read_tile_sizes,
    read_tile_transformations,
)

PROJ_AXIS = {"xy": 2, "xz": 1, "yz": 0}  # pragma: no cover
AXIS_PROJ = {2: "xy", 1: "xz", 0: "yz"}  # pragma: no cover
PROJ_KEEP = {2: np.array([0, 1]), 1: np.array([0, 2]), 0: np.array([1, 2])}  # pragma: no cover

LOGGER = logging.getLogger("ip_density")  # pragma: no cover


def plot_pair_IP_density(
    tile1: int,
    tile2: int,
    ip_arrays,
    tile_transformations,
    tile_inv_transformations,
    tile_sizes,
    ip_correspondences=None,
    id_maps=None,
    corresponding_only=False,
    pdf_writer=None,
):  # pragma: no cover
    """Create a plot of the tile1-tile2 boundary IP density."""
    title_mode = "all"
    w_box_overlap, _, _ = get_tile_pair_overlap(
        tile1, tile2, tile_transformations, tile_inv_transformations, tile_sizes
    )
    if w_box_overlap is None:
        LOGGER.info(f"Tile {tile1} and tile {tile2} do not overlap.")
        return
    LOGGER.info(f"Tile {tile1} and tile {tile2} overlap in world coordinates: {w_box_overlap}")
    ips1 = get_tile_overlapping_IPs(
        tile1, tile2, ip_arrays[tile1], tile_transformations, tile_inv_transformations, tile_sizes
    )
    ips2 = get_tile_overlapping_IPs(
        tile2, tile1, ip_arrays[tile2], tile_transformations, tile_inv_transformations, tile_sizes
    )
    if corresponding_only:
        ips1 = filter_tile_corresponding_IPs(tile1, tile2, ips1, ip_correspondences, id_maps)
        ips2 = filter_tile_corresponding_IPs(tile2, tile1, ips2, ip_correspondences, id_maps)
        title_mode = "corresp."

    fig = plt.figure(figsize=(6, 12))
    for proj_axis in (0, 1, 2):
        ax = fig.add_subplot(3, 2, 2 * proj_axis + 1)

        nbins = (
            int(w_box_overlap.tright[PROJ_KEEP[proj_axis]][0] - w_box_overlap.bleft[PROJ_KEEP[proj_axis]][0])
            // 200,
            int(w_box_overlap.tright[PROJ_KEEP[proj_axis]][1] - w_box_overlap.bleft[PROJ_KEEP[proj_axis]][1])
            // 200,
        )

        coords1 = ips1["loc_w"][:, PROJ_KEEP[proj_axis]]
        H, xedges, yedges = np.histogram2d(coords1[:, 0], coords1[:, 1], bins=nbins)
        vmax1 = np.max(H)

        coords2 = ips2["loc_w"][:, PROJ_KEEP[proj_axis]]
        H, xedges, yedges = np.histogram2d(coords2[:, 0], coords2[:, 1], bins=nbins)
        vmax2 = np.max(H)

        vmax = max(vmax1, vmax2)

        H = ax.hist2d(
            coords1[:, 0],
            coords1[:, 1],
            range=[
                [w_box_overlap.bleft[PROJ_KEEP[proj_axis]][0], w_box_overlap.tright[PROJ_KEEP[proj_axis]][0]],
                [w_box_overlap.bleft[PROJ_KEEP[proj_axis]][1], w_box_overlap.tright[PROJ_KEEP[proj_axis]][1]],
            ],
            vmin=0,
            vmax=vmax,
            bins=nbins,
            cmap="Blues",
        )
        ax.set_xlim(
            w_box_overlap.bleft[PROJ_KEEP[proj_axis]][0], w_box_overlap.tright[PROJ_KEEP[proj_axis]][0]
        )
        ax.set_ylim(
            w_box_overlap.bleft[PROJ_KEEP[proj_axis]][1], w_box_overlap.tright[PROJ_KEEP[proj_axis]][1]
        )
        ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(format_large_numbers))
        ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(format_large_numbers))
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_title(f"T{tile1} in {AXIS_PROJ[proj_axis]}")
        fig.colorbar(H[3], ax=ax)

        ax = fig.add_subplot(3, 2, 2 * proj_axis + 2)
        H = ax.hist2d(
            coords2[:, 0],
            coords2[:, 1],
            range=[
                [w_box_overlap.bleft[PROJ_KEEP[proj_axis]][0], w_box_overlap.tright[PROJ_KEEP[proj_axis]][0]],
                [w_box_overlap.bleft[PROJ_KEEP[proj_axis]][1], w_box_overlap.tright[PROJ_KEEP[proj_axis]][1]],
            ],
            vmin=0,
            vmax=vmax,
            bins=nbins,
            cmap="Blues",
        )
        ax.set_xlim(
            w_box_overlap.bleft[PROJ_KEEP[proj_axis]][0], w_box_overlap.tright[PROJ_KEEP[proj_axis]][0]
        )
        ax.set_ylim(
            w_box_overlap.bleft[PROJ_KEEP[proj_axis]][1], w_box_overlap.tright[PROJ_KEEP[proj_axis]][1]
        )
        ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(format_large_numbers))
        ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(format_large_numbers))
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_title(f"T{tile2} in {AXIS_PROJ[proj_axis]}")

        fig.colorbar(H[3], ax=ax)
    fig.suptitle(f"Tile{tile1}-{tile2} overlap ({title_mode})")
    if pdf_writer:
        pdf_writer.savefig(fig)
        plt.close(fig)


def run_density_plot(input_xml: str, prefix: Optional[str] = None):  # pragma: no cover
    """Extract overlap from tile pairs

    interestpoints.n5 must be in the current working directory.
    """
    with open(input_xml) as f:
        xmldict = xmltodict.parse(f.read())
    # get all transformations
    ip_arrays = read_tiles_interestpoints()
    ip_correspondences, id_maps = read_ip_correspondences()
    tile_sizes = read_tile_sizes(xmldict["SpimData"]["SequenceDescription"]["ViewSetups"])
    tile_transformations, tile_inv_transformations = read_tile_transformations(
        xmldict["SpimData"]["ViewRegistrations"]
    )
    vertical_pairs = [
        (0, 3),
        (1, 4),
        (2, 5),
        (3, 6),
        (4, 7),
        (5, 8),
        (6, 9),
        (7, 10),
        (8, 11),
        (9, 12),
        (10, 13),
        (11, 14),
    ]
    horizontal_pairs = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (9, 10), (10, 11), (12, 13), (13, 14)]
    if not prefix:
        prefix = ""
    with PdfPages(f"{prefix}ip_density_vertical_overlaps.pdf") as pdf_writer:
        for t1, t2 in vertical_pairs:
            plot_pair_IP_density(
                t1,
                t2,
                ip_arrays,
                tile_transformations,
                tile_inv_transformations,
                tile_sizes,
                ip_correspondences=ip_correspondences,
                id_maps=id_maps,
                corresponding_only=False,
                pdf_writer=pdf_writer,
            )
            plot_pair_IP_density(
                t1,
                t2,
                ip_arrays,
                tile_transformations,
                tile_inv_transformations,
                tile_sizes,
                ip_correspondences=ip_correspondences,
                id_maps=id_maps,
                corresponding_only=True,
                pdf_writer=pdf_writer,
            )

    with PdfPages(f"{prefix}ip_density_horizontal_overlaps.pdf") as pdf_writer:
        for t1, t2 in horizontal_pairs:
            plot_pair_IP_density(
                t1,
                t2,
                ip_arrays,
                tile_transformations,
                tile_inv_transformations,
                tile_sizes,
                ip_correspondences=ip_correspondences,
                id_maps=id_maps,
                corresponding_only=False,
                pdf_writer=pdf_writer,
            )
            plot_pair_IP_density(
                t1,
                t2,
                ip_arrays,
                tile_transformations,
                tile_inv_transformations,
                tile_sizes,
                ip_correspondences=ip_correspondences,
                id_maps=id_maps,
                corresponding_only=True,
                pdf_writer=pdf_writer,
            )


def run_tr_density_plot():  # pragma: no cover
    """Entry point for run_tr_density_plot."""
    rlogger = logging.getLogger()
    rlogger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    rlogger.addHandler(handler)
    run_density_plot("../results/bigstitcher.xml", prefix="../results/tr_")


def run_aff_density_plot():  # pragma: no cover
    """Entry point for run_aff_density_plot."""
    rlogger = logging.getLogger()
    rlogger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    rlogger.addHandler(handler)
    run_density_plot("../results/bigstitcher.xml", prefix="../results/aff_")

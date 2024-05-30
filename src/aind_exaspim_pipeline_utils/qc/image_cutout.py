"""Create transformed output images from input images by scipy.ndimage.affine_transform."""
import logging
import sys
from typing import Optional, Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xmltodict

from .bbox import Bbox
from .tile_transformations import (
    get_transformed_pair_cutouts,
    read_tile_sizes,
    read_tile_transformations,
    get_tile_overlapping_IPs,
    filter_tile_corresponding_IPs,
    format_large_numbers,
    read_tiles_interestpoints,
    read_ip_correspondences,
)
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker


PROJ_AXIS = {"xy": 2, "xz": 1, "yz": 0}  # pragma: no cover
AXIS_PROJ = {2: "xy", 1: "xz", 0: "yz"}  # pragma: no cover
PROJ_KEEP = {2: np.array([0, 1]), 1: np.array([0, 2]), 0: np.array([1, 2])}  # pragma: no cover

LOGGER = logging.getLogger("simple_cutout")  # pragma: no cover


def create_pair_overplot(
    t1: int,
    t2: int,
    t1_cutout: np.ndarray,
    t2_cutout: np.ndarray,
    w_box_overlap: Bbox,
    pdf_writer: Optional[PdfPages] = None,
    common_scale: bool = False,
):  # pragma: no cover
    """Create a plot of the t1-t2 boundary with the transformed images (t1, t2, overplot)."""
    vmin1 = np.percentile(t1_cutout, 1)
    vmin2 = np.percentile(t2_cutout, 1)
    vmin = min(vmin1, vmin2)
    vmax1 = np.percentile(t1_cutout, 99)
    if vmax1 < 1:
        vmax1 = 1
    vmax2 = np.percentile(t2_cutout, 99)
    if vmax2 < 1:
        vmax2 = 1
    vmax = max(vmax1, vmax2)
    if common_scale:
        vmin1 = vmin
        vmin2 = vmin
        vmax1 = vmax
        vmax2 = vmax

    LOGGER.info(f"vmin1 = {vmin1}, vmax1 = {vmax1}")
    LOGGER.info(f"vmin2 = {vmin2}, vmax2 = {vmax2}")
    # plot
    fig = plt.figure(figsize=(6, 12))
    for proj_axis in (0, 1, 2):
        mips_t1 = np.amax(t1_cutout, axis=proj_axis)
        mips_t2 = np.amax(t2_cutout, axis=proj_axis)

        ax = fig.add_subplot(3, 3, 3 * proj_axis + 1)
        ax.imshow(
            mips_t1.transpose(),
            cmap="gray",
            vmin=vmin1,
            vmax=vmax1,
            interpolation="none",
            extent=[
                w_box_overlap.bleft[PROJ_KEEP[proj_axis]][0] - 0.5,
                w_box_overlap.tright[PROJ_KEEP[proj_axis]][0] - 0.5,
                w_box_overlap.tright[PROJ_KEEP[proj_axis]][1] - 0.5,
                w_box_overlap.bleft[PROJ_KEEP[proj_axis]][1] - 0.5,
            ],
        )
        # ax.set_aspect("equal")
        ax.set_title(f"T{t1} in {AXIS_PROJ[proj_axis]}")
        ax = fig.add_subplot(3, 3, 3 * proj_axis + 2)
        ax.imshow(
            mips_t2.transpose(),
            cmap="gray",
            vmin=vmin2,
            vmax=vmax2,
            interpolation="none",
            extent=[
                w_box_overlap.bleft[PROJ_KEEP[proj_axis]][0] - 0.5,
                w_box_overlap.tright[PROJ_KEEP[proj_axis]][0] - 0.5,
                w_box_overlap.tright[PROJ_KEEP[proj_axis]][1] - 0.5,
                w_box_overlap.bleft[PROJ_KEEP[proj_axis]][1] - 0.5,
            ],
        )
        ax.set_aspect("equal")
        ax.set_title(f"T{t2} in {AXIS_PROJ[proj_axis]}")

        ax = fig.add_subplot(3, 3, 3 * proj_axis + 3)
        mips_rgb = np.zeros((mips_t1.shape[1], mips_t1.shape[0], 3), dtype=float)
        A = np.maximum([[0.0]], mips_t1.transpose() - vmin1)
        A /= vmax1
        A = np.minimum([[1.0]], A)
        mips_rgb[:, :, 0] = A
        A = np.maximum([[0.0]], mips_t2.transpose() - vmin2)
        A /= vmax2
        A = np.minimum([[1.0]], A)
        mips_rgb[:, :, 1] = A

        ax.imshow(
            mips_rgb,
            interpolation="none",
            extent=[
                w_box_overlap.bleft[PROJ_KEEP[proj_axis]][0] - 0.5,
                w_box_overlap.tright[PROJ_KEEP[proj_axis]][0] - 0.5,
                w_box_overlap.tright[PROJ_KEEP[proj_axis]][1] - 0.5,
                w_box_overlap.bleft[PROJ_KEEP[proj_axis]][1] - 0.5,
            ],
        )
        ax.set_aspect("equal")
        ax.set_title(f"T{t1} + T{t2} in {AXIS_PROJ[proj_axis]}")
    if pdf_writer:
        pdf_writer.savefig(fig)
        plt.close(fig)


def plot_one_combined_projection(
    tile1: int,
    tile2: int,
    ips1: np.ndarray,
    ips2: np.ndarray,
    t1_cutout: np.ndarray,
    t2_cutout: np.ndarray,
    w_box_overlap: Bbox,
    proj_axis: int,
    axs: Iterable[plt.Axes],
    fig: plt.Figure,
    common_scale: bool = False,
):  # pragma: no cover
    """Plots one projection of tile1-tile2 boundary IP density and transformed images.

    Parameters
    ----------
    tile1, tile2 : int
        Tile numbers.
    ips1, ips2 : Structured np.ndarray
        Arrays of interest points to show. Either the all of them
        in the overlap area or just the corresponding ones.
    t1_cutout, t2_cutout : np.ndarray
        Cutouts of the transformed tiles.
    w_box_overlap : Bbox
        The boundary box of the overlap area in world coordinates.
    proj_axis : int
        The projection axis.
    ax : plt.Axes
        The axis to plot on.
    fig : plt.Figure
        The figure to plot on. Used for colorbar repositioning.
    common_scale : bool
        Whether to use a common value range in the color scale for the images. If not, colorbar is not shown.
    """
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

    mips_t1 = np.amax(t1_cutout, axis=proj_axis)
    mips_t2 = np.amax(t2_cutout, axis=proj_axis)

    # Determine the image cutouts value ranges

    img_vmin1 = np.percentile(t1_cutout, 1)
    img_vmin2 = np.percentile(t2_cutout, 1)
    img_vmin = min(img_vmin1, img_vmin2)
    img_vmax1 = np.percentile(t1_cutout, 99)
    if img_vmax1 < 1:
        img_vmax1 = 1
    img_vmax2 = np.percentile(t2_cutout, 99)
    if img_vmax2 < 1:
        img_vmax2 = 1
    img_vmax = max(img_vmax1, img_vmax2)
    if common_scale:
        img_vmin1 = img_vmin
        img_vmin2 = img_vmin
        img_vmax1 = img_vmax
        img_vmax2 = img_vmax

    ax_iter = iter(axs)
    ax = next(ax_iter)
    ax.imshow(
        mips_t1.transpose(),
        cmap="gray",
        vmin=img_vmin1,
        vmax=img_vmax1,
        interpolation="none",
        extent=[
            w_box_overlap.bleft[PROJ_KEEP[proj_axis]][0] - 0.5,
            w_box_overlap.tright[PROJ_KEEP[proj_axis]][0] - 0.5,
            w_box_overlap.tright[PROJ_KEEP[proj_axis]][1] - 0.5,
            w_box_overlap.bleft[PROJ_KEEP[proj_axis]][1] - 0.5,
        ],
    )

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
        alpha=0.9,
    )
    # ax.set_xlim(w_box_overlap.bleft[PROJ_KEEP[proj_axis]][0], w_box_overlap.tright[PROJ_KEEP[proj_axis]][0])
    # ax.set_ylim(w_box_overlap.bleft[PROJ_KEEP[proj_axis]][1], w_box_overlap.tright[PROJ_KEEP[proj_axis]][1])
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(format_large_numbers))
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(format_large_numbers))
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"T{tile1} in {AXIS_PROJ[proj_axis]}")
    fig.colorbar(H[3], ax=ax)

    # Middle panel the overlap cutout of the tiles
    ax = next(ax_iter)

    # Create the RGB image of the two cutouts
    mips_rgb = np.zeros((mips_t1.shape[1], mips_t1.shape[0], 3), dtype=float)
    A = np.maximum([[0.0]], mips_t1.transpose() - img_vmin1)
    A /= img_vmax1
    A = np.minimum([[1.0]], A)
    mips_rgb[:, :, 0] = A
    A = np.maximum([[0.0]], mips_t2.transpose() - img_vmin2)
    A /= img_vmax2
    A = np.minimum([[1.0]], A)
    mips_rgb[:, :, 1] = A

    ax.imshow(
        mips_rgb,
        interpolation="none",
        extent=[
            w_box_overlap.bleft[PROJ_KEEP[proj_axis]][0] - 0.5,
            w_box_overlap.tright[PROJ_KEEP[proj_axis]][0] - 0.5,
            w_box_overlap.tright[PROJ_KEEP[proj_axis]][1] - 0.5,
            w_box_overlap.bleft[PROJ_KEEP[proj_axis]][1] - 0.5,
        ],
    )
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(format_large_numbers))
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(format_large_numbers))
    if common_scale:
        # Create a gradient from black (0, 0, 0) to red (255, 0, 0)
        gradient = np.linspace(0, 1, 256)
        colors = np.vstack((gradient, np.zeros(256), np.zeros(256))).T

        # Create a custom colormap
        custom_cmap = plt.cm.colors.ListedColormap(colors)
        fig.colorbar(
            matplotlib.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(img_vmin, img_vmax), cmap=custom_cmap
            ),
            ax=ax,
        )
    ax.set_aspect("equal")
    ax.set_title(f"T{tile1} + T{tile2} in {AXIS_PROJ[proj_axis]}")

    # Right panel, ip density on tile2
    ax = next(ax_iter)

    ax.imshow(
        mips_t1.transpose(),
        cmap="gray",
        vmin=img_vmin1,
        vmax=img_vmax1,
        interpolation="none",
        extent=[
            w_box_overlap.bleft[PROJ_KEEP[proj_axis]][0] - 0.5,
            w_box_overlap.tright[PROJ_KEEP[proj_axis]][0] - 0.5,
            w_box_overlap.tright[PROJ_KEEP[proj_axis]][1] - 0.5,
            w_box_overlap.bleft[PROJ_KEEP[proj_axis]][1] - 0.5,
        ],
    )
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
        alpha=0.9,
    )
    # ax.set_xlim(w_box_overlap.bleft[PROJ_KEEP[proj_axis]][0], w_box_overlap.tright[PROJ_KEEP[proj_axis]][0])
    # ax.set_ylim(w_box_overlap.bleft[PROJ_KEEP[proj_axis]][1], w_box_overlap.tright[PROJ_KEEP[proj_axis]][1])
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(format_large_numbers))
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(format_large_numbers))
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"T{tile2} in {AXIS_PROJ[proj_axis]}")
    fig.colorbar(H[3], ax=ax)


def create_one_projection_combined_figure(
    tile1: int,
    tile2: int,
    ip_arrays,
    tile_transformations,
    tile_inv_transformations,
    tile_sizes,
    w_box_overlap: Bbox,
    t1_cutout,
    t2_cutout,
    ip_correspondences=None,
    id_maps=None,
    corresponding_only=False,
    pdf_writer=None,
    common_scale: bool = False,
    proj_axis: Optional[int] = 0,
):  # pragma: no cover
    """Create a plot of the tile1-tile2 boundary IP density and include the transformed images."""
    title_mode = "all"

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

    if proj_axis is None:
        fig = plt.figure(figsize=(10, 12))
        for proj_axis in (0, 1, 2):
            ax1 = fig.add_subplot(3, 3, 3 * proj_axis + 1)
            ax2 = fig.add_subplot(3, 3, 3 * proj_axis + 2)
            ax3 = fig.add_subplot(3, 3, 3 * proj_axis + 3)
            plot_one_combined_projection(
                tile1,
                tile2,
                ips1,
                ips2,
                t1_cutout,
                t2_cutout,
                w_box_overlap,
                proj_axis,
                [ax1, ax2, ax3],
                fig,
                common_scale,
            )
    else:
        fig, axs = plt.subplots(1, 3, figsize=(15, 11))
        plot_one_combined_projection(
            tile1, tile2, ips1, ips2, t1_cutout, t2_cutout, w_box_overlap, proj_axis, axs, fig, common_scale
        )
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90)
    fig.suptitle(f"Tile{tile1}-{tile2} overlap ({title_mode})")
    if pdf_writer:
        pdf_writer.savefig(fig)
        plt.close(fig)


def run_pair_overplots(input_xml: str, prefix: Optional[str] = None):  # pragma: no cover
    """Create image cutout plots with tile1, tile2, overplot
    for all the vertical and horizontal tile pairs."""
    with open(input_xml) as f:
        xmldict = xmltodict.parse(f.read())
    # get all transformations
    tile_full_sizes = read_tile_sizes(xmldict["SpimData"]["SequenceDescription"]["ViewSetups"])
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
    with PdfPages(f"{prefix}cutouts_vertical_overlaps.pdf") as pdf_writer:
        for t1, t2 in vertical_pairs:
            t1_cutout, t2_cutout, w_box_overlap = get_transformed_pair_cutouts(
                t1, t2, 4, tile_transformations, tile_inv_transformations, tile_full_sizes, xmldict
            )
            create_pair_overplot(
                t1, t2, t1_cutout, t2_cutout, w_box_overlap, common_scale=True, pdf_writer=pdf_writer
            )
    with PdfPages(f"{prefix}cutouts_horizontal_overlaps.pdf") as pdf_writer:
        for t1, t2 in horizontal_pairs:
            t1_cutout, t2_cutout, w_box_overlap = get_transformed_pair_cutouts(
                t1, t2, 4, tile_transformations, tile_inv_transformations, tile_full_sizes, xmldict
            )
            create_pair_overplot(
                t1, t2, t1_cutout, t2_cutout, w_box_overlap, common_scale=True, pdf_writer=pdf_writer
            )


def run_combined_plots(
    input_xml: str,
    prefix: Optional[str] = None,
    vert_proj_axis: Optional[int] = None,
    hor_proj_axis: Optional[int] = None,
):  # pragma: no cover
    """Create IP density and image cutout plots for all the vertical and horizontal tile pairs.

    Parameters
    ----------
    input_xml : str
        The path to the BigStitcher XML file.
    prefix : str, optional
        The prefix to add to the output file names.
    vert_proj_axis, hor_proj_axis : int, optional
        The projection axis to use for the vertical and horizontal overlaps respectively.
        If None, all three projections are plotted.
    """
    with open(input_xml) as f:
        xmldict = xmltodict.parse(f.read())
    # get the interest points
    ip_arrays = read_tiles_interestpoints()
    ip_correspondences, id_maps = read_ip_correspondences()
    # get all transformations
    tile_full_sizes = read_tile_sizes(xmldict["SpimData"]["SequenceDescription"]["ViewSetups"])
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
    if vert_proj_axis is None:
        pdf_proj = ""
    else:
        pdf_proj = "_" + AXIS_PROJ[vert_proj_axis]
    with PdfPages(f"{prefix}cutouts_vertical_overlaps{pdf_proj}.pdf") as pdf_writer:
        for t1, t2 in vertical_pairs:
            t1_cutout, t2_cutout, w_box_overlap = get_transformed_pair_cutouts(
                t1, t2, 4, tile_transformations, tile_inv_transformations, tile_full_sizes, xmldict
            )
            if w_box_overlap is None:
                LOGGER.warning(f"Tile {t1} and {t2} has world overlap box. Skipping.")
                continue
            create_one_projection_combined_figure(
                t1,
                t2,
                ip_arrays,
                tile_transformations,
                tile_inv_transformations,
                tile_full_sizes,
                w_box_overlap,
                t1_cutout,
                t2_cutout,
                ip_correspondences,
                id_maps,
                corresponding_only=False,
                pdf_writer=pdf_writer,
                common_scale=True,
                proj_axis=vert_proj_axis,
            )
            create_one_projection_combined_figure(
                t1,
                t2,
                ip_arrays,
                tile_transformations,
                tile_inv_transformations,
                tile_full_sizes,
                w_box_overlap,
                t1_cutout,
                t2_cutout,
                ip_correspondences,
                id_maps,
                corresponding_only=True,
                pdf_writer=pdf_writer,
                common_scale=False,
                proj_axis=vert_proj_axis,
            )
    if hor_proj_axis is None:
        pdf_proj = ""
    else:
        pdf_proj = "_" + AXIS_PROJ[hor_proj_axis]
    with PdfPages(f"{prefix}cutouts_horizontal_overlaps{pdf_proj}.pdf") as pdf_writer:
        for t1, t2 in horizontal_pairs:
            t1_cutout, t2_cutout, w_box_overlap = get_transformed_pair_cutouts(
                t1, t2, 4, tile_transformations, tile_inv_transformations, tile_full_sizes, xmldict
            )
            if w_box_overlap is None:
                LOGGER.warning(f"Tile {t1} and {t2} has world overlap box. Skipping.")
                continue

            create_one_projection_combined_figure(
                t1,
                t2,
                ip_arrays,
                tile_transformations,
                tile_inv_transformations,
                tile_full_sizes,
                w_box_overlap,
                t1_cutout,
                t2_cutout,
                ip_correspondences,
                id_maps,
                corresponding_only=False,
                pdf_writer=pdf_writer,
                common_scale=True,
                proj_axis=hor_proj_axis,
            )
            create_one_projection_combined_figure(
                t1,
                t2,
                ip_arrays,
                tile_transformations,
                tile_inv_transformations,
                tile_full_sizes,
                w_box_overlap,
                t1_cutout,
                t2_cutout,
                ip_correspondences,
                id_maps,
                corresponding_only=True,
                pdf_writer=pdf_writer,
                common_scale=False,
                proj_axis=hor_proj_axis,
            )


def run_aff_cutout_plot():  # pragma: no cover
    """Entry point for run_aff_cutout_plot."""
    rlogger = logging.getLogger()
    rlogger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    rlogger.addHandler(handler)
    run_pair_overplots("../results/bigstitcher.xml", prefix="../results/aff_")


def run_aff_combined_plot():  # pragma: no cover
    """Entry point for run_aff_combined_plot."""
    rlogger = logging.getLogger()
    rlogger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    rlogger.addHandler(handler)
    run_combined_plots("../results/bigstitcher.xml", prefix="../results/aff_")


def run_aff_yz_xz_combined_plot():  # pragma: no cover
    """Entry point for run_aff_yz_xz_combined_plot."""
    rlogger = logging.getLogger()
    rlogger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    rlogger.addHandler(handler)
    run_combined_plots(
        "../results/bigstitcher.xml", prefix="../results/aff_", vert_proj_axis=0, hor_proj_axis=1
    )

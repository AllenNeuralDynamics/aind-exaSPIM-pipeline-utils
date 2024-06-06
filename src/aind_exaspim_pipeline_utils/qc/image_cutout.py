"""Create transformed output images from input images by scipy.ndimage.affine_transform."""
import logging
import sys
from collections import OrderedDict
from typing import Optional, Iterable, Dict, Tuple

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
from .spimdata import SplitImageLoader, ZarrImageLoader
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

def determine_data_vmin_vmax(data: Iterable[np.ndarray], percentile_cut: bool = True):
    """Determine the value range for the images and histograms.

    Assume positive values only. If the minimum value is negative, it is set to 0.
    If the maximum value is less than 1, it is set to 1.

    Parameters
    ----------
    data : Iterable[np.ndarray]
        The image or histogram data to determine the common value range for.
    percentile_cut : bool
        Whether to use the 1st and 99th percentiles as the value range. If False,
        the minimum and maximum values are used.

    Returns
    -------
    vmin, vmax : float
        The common value range for the images and histograms. Default is 0 to 1.
    """
    first = True
    for img in data:
        if first:
            if percentile_cut:
                vmin = np.percentile(img, 1)
                vmax = np.percentile(img, 99)
            else:
                vmin = np.min(img)
                vmax = np.max(img)
            first = False
        else:
            if percentile_cut:
                vmin = min(vmin, np.percentile(img, 1))
                vmax = max(vmax, np.percentile(img, 99))
            else:
                vmin = min(vmin, np.min(img))
                vmax = max(vmax, np.max(img))
    if first:
        vmin = 0
        vmax = 1
    if vmin < 0:
        vmin = 0
    if vmax < 1:
        vmax = 1
    return vmin, vmax


def plot_one_panel_trio(
    tile1: int,
    tile2: int,
    ips1: Optional[np.ndarray],
    ips2: Optional[np.ndarray],
    mips_t1: np.ndarray,
    mips_t2: np.ndarray,
    w_box_overlap: Bbox,
    proj_axis: int,
    axs: Iterable[plt.Axes],
    fig: plt.Figure,
    img_vmin1: float,
    img_vmax1: float,
    img_vmin2: float,
    img_vmax2: float,
    hist_vmax1: float,
    hist_vmax2: float,
    common_scale: bool = False,
    subtile_plot: bool = False,
):  # pragma: no cover
    """Plots one trio of (sub)panels with IP density and transformed images.

    If plotting outer tiles, the colorbars are drawn. If plotting subtiles, the colorbar mappables
    are returned.

    Parameters
    ----------
    tile1, tile2 : int
        Tile numbers. Can be the outer tile numbers or the subtiles.
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

    # if ips1 is not None:
    #     coords1 = ips1["loc_w"][:, PROJ_KEEP[proj_axis]]
    #     H, xedges, yedges = np.histogram2d(coords1[:, 0], coords1[:, 1], bins=nbins)
    #     vmax1 = np.max(H)
    # else:
    #     vmax1 = 1

    # if ips2 is not None:
    #     coords2 = ips2["loc_w"][:, PROJ_KEEP[proj_axis]]
    #     H, xedges, yedges = np.histogram2d(coords2[:, 0], coords2[:, 1], bins=nbins)
    #     vmax2 = np.max(H)
    # else:
    #     vmax2 = 1

    # vmax = max(vmax1, vmax2)

    # mips_t1 = np.amax(t1_cutout, axis=proj_axis)
    # mips_t2 = np.amax(t2_cutout, axis=proj_axis)

    # Determine the image cutouts value ranges

    # img_vmin1 = np.percentile(t1_cutout, 1)
    # img_vmin2 = np.percentile(t2_cutout, 1)
    # img_vmin = min(img_vmin1, img_vmin2)
    # img_vmax1 = np.percentile(t1_cutout, 99)
    # if img_vmax1 < 1:
    #     img_vmax1 = 1
    # img_vmax2 = np.percentile(t2_cutout, 99)
    # if img_vmax2 < 1:
    #     img_vmax2 = 1
    # img_vmax = max(img_vmax1, img_vmax2)
    if common_scale:
        img_vmin = min(img_vmin1, img_vmin2)
        img_vmax = max(img_vmax1, img_vmax2)
        img_vmin1 = img_vmin
        img_vmin2 = img_vmin
        img_vmax1 = img_vmax
        img_vmax2 = img_vmax

    cbar_mappable = []
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

    if ips1 is not None:
        coords1 = ips1["loc_w"][:, PROJ_KEEP[proj_axis]]
        H = ax.hist2d(
            coords1[:, 0],
            coords1[:, 1],
            range=[
                [w_box_overlap.bleft[PROJ_KEEP[proj_axis]][0], w_box_overlap.tright[PROJ_KEEP[proj_axis]][0]],
                [w_box_overlap.bleft[PROJ_KEEP[proj_axis]][1], w_box_overlap.tright[PROJ_KEEP[proj_axis]][1]],
            ],
            vmin=0,
            vmax=hist_vmax1,
            bins=nbins,
            cmap="Blues",
            alpha=0.9,
        )
        ax.invert_yaxis()
        cbar_mappable.append(H[3])
    else:
        cbar_mappable.append(None)

    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(format_large_numbers))
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(format_large_numbers))

    if ips1 is not None and not subtile_plot:
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
        if subtile_plot:
            cbar_mappable.append(
                matplotlib.cm.ScalarMappable(
                    norm=matplotlib.colors.Normalize(img_vmin, img_vmax), cmap=custom_cmap
                )
            )
        else:
            fig.colorbar(
                matplotlib.cm.ScalarMappable(
                    norm=matplotlib.colors.Normalize(img_vmin, img_vmax), cmap=custom_cmap
                ),
                ax=ax,
            )
            cbar_mappable.append(None)
    else:
        cbar_mappable.append(None)
    if not subtile_plot:
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
    if ips2 is not None:
        coords2 = ips2["loc_w"][:, PROJ_KEEP[proj_axis]]
        H = ax.hist2d(
            coords2[:, 0],
            coords2[:, 1],
            range=[
                [w_box_overlap.bleft[PROJ_KEEP[proj_axis]][0], w_box_overlap.tright[PROJ_KEEP[proj_axis]][0]],
                [w_box_overlap.bleft[PROJ_KEEP[proj_axis]][1], w_box_overlap.tright[PROJ_KEEP[proj_axis]][1]],
            ],
            vmin=0,
            vmax=hist_vmax2,
            bins=nbins,
            cmap="Blues",
            alpha=0.9,
        )
        ax.invert_yaxis()
        cbar_mappable.append(H[3])
    else:
        cbar_mappable.append(None)
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(format_large_numbers))
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(format_large_numbers))
    if ips2 is not None and not subtile_plot:
        ax.set_aspect("equal")
        ax.set_title(f"T{tile2} in {AXIS_PROJ[proj_axis]}")
        fig.colorbar(H[3], ax=ax)
    return cbar_mappable


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
            plot_one_panel_trio(
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
        plot_one_panel_trio(
            tile1, tile2, ips1, ips2, t1_cutout, t2_cutout, w_box_overlap, proj_axis, axs, fig, common_scale
        )
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90)
    fig.suptitle(f"Tile{tile1}-{tile2} overlap ({title_mode})")
    if pdf_writer:
        pdf_writer.savefig(fig)
        plt.close(fig)

def get_subtile_overlapping_IPs(st_pairs: Iterable[int,int], ip_arrays, tile_transformations, tile_inv_transformations, tile_sizes,
                                ip_correspondences=None, id_maps=None, corresponding_only=False):
    """Get the interest points of tile pairs. If corresponding_only is True, only the corresponding IPs are returned.

    Returns
    -------
    st_ips : OrderedDict[(int,int), Tuple[np.ndarray, np.ndarray]]
        The interest points of the tile pairs. The first element of the tuple is the IPs of the first tile,
        the second element is the IPs of the second tile. If the tile does not have loaded IPs, the element is None.
    """
    st_ips = OrderedDict()
    for (st1, st2) in st_pairs:
        if st1 in ip_arrays:
            ips1 = get_tile_overlapping_IPs(
                st1, st2, ip_arrays[st1], tile_transformations, tile_inv_transformations, tile_sizes
            )
        else:
            ips1 = None

        if st2 in ip_arrays:
            ips2 = get_tile_overlapping_IPs(
                st2, st1, ip_arrays[st2], tile_transformations, tile_inv_transformations, tile_sizes
            )
        else:
            ips2 = None
        if corresponding_only:
            if ips1:
                ips1 = filter_tile_corresponding_IPs(st1, st2, ips1, ip_correspondences, id_maps)
            if ips2:
                ips2 = filter_tile_corresponding_IPs(st2, st1, ips2, ip_correspondences, id_maps)

        st_ips[(st1,st2)] = ips1, ips2
    return st_ips

def get_histograms_vmin_vmax(st_pairs: Iterable[int,int], st_ips: Dict[int, np.ndarray],
                             st_overlaps: Dict[Tuple[int, int], Bbox], proj_axis: int = 0,
                             common_scale: bool = False):
    """
    Determine the minimum and maximum values for the left side subpanels (st1-s in st_pairs) and
    the right side subpanels (st2-s in st_pairs).

    Returns
    -------
    st_hist_vmax1, st_hist_vmax2 : float
        The maximum values of the left and right side subpanels.
    """
    st_hist_vmax1 = 1
    st_hist_vmax2 = 1
    for st1, st2 in st_pairs:
        if (st1, st2) in st_overlaps:
            w_box_overlap = st_overlaps[(st1, st2)]
            nbins = (
           int(w_box_overlap.tright[PROJ_KEEP[proj_axis]][0] - w_box_overlap.bleft[PROJ_KEEP[proj_axis]][0]) // 200,
           int(w_box_overlap.tright[PROJ_KEEP[proj_axis]][1] - w_box_overlap.bleft[PROJ_KEEP[proj_axis]][1]) // 200,
            )
            if st1 in st_ips:
                coords1 = st_ips[st1]["loc_w"][:, proj_axis]
                H, xedges, yedges = np.histogram2d(coords1[:, 0], coords1[:, 1], bins=nbins)
                st_hist_vmax1 = max(st_hist_vmax1, np.max(H))
            if st2 in st_ips:
                coords2 = st_ips[st2]["loc_w"][:, proj_axis]
                H, xedges, yedges = np.histogram2d(coords2[:, 0], coords2[:, 1], bins=nbins)
                st_hist_vmax2 = max(st_hist_vmax2, np.max(H))
    if common_scale:
        hist_vmax = max(st_hist_vmax1, st_hist_vmax2)
        st_hist_vmax1 = hist_vmax
        st_hist_vmax2 = hist_vmax
    return st_hist_vmax1, st_hist_vmax2
def get_subtile_mips_and_values(st_pairs: Iterable[int,int], st_cutouts: Dict[int, np.ndarray], common_scale: bool = False,
                                proj_axis: int = 0):
    """
    Determine the minimum and maximum values for the left side subpanels (st1-s in st_pairs) and
    the right side subpanels (st2-s in st_pairs).

    Returns
    -------
    st_mips : OrderedDict[int, np.ndarray]
        The maximum intensity projections of all the subpanels. First the left sides, then the right sides.
    img_vmin1, img_vmax1, img_vmin2, img_vmax2 : float
        The minimum and maximum values of the left and right side subpanels. If common_scale is True,
        the minimum and maximum values are the same for all subpanels.
    """
    st_mips1 = OrderedDict()
    st_mips2 = OrderedDict()
    for st1, st2 in st_pairs:
        mips_t1 = np.amax(st_cutouts[st1], axis=proj_axis)
        mips_t2 = np.amax(st_cutouts[st2], axis=proj_axis)
        st_mips1[st1]=mips_t1
        st_mips2[st2]=mips_t2
    img_vmin1, img_vmax1 = determine_data_vmin_vmax(st_mips1, percentile_cut=True)
    img_vmin2, img_vmax2 = determine_data_vmin_vmax(st_mips2, percentile_cut=True)
    if common_scale:
        img_vmin = min(img_vmin1, img_vmin2)
        img_vmax = max(img_vmax1, img_vmax2)
        img_vmin1 = img_vmin
        img_vmin2 = img_vmin
        img_vmax1 = img_vmax
        img_vmax2 = img_vmax
    st_mips1.update(st_mips2)
    return st_mips1, img_vmin1, img_vmax1, img_vmin2, img_vmax2

def create_one_projection_split_tiles_figure(
    outer_tile1: int,
    outer_tile2: int,
    ip_arrays,
    tile_transformations,
    tile_inv_transformations,
    tile_sizes,
    st_overlaps: Dict[Tuple[int, int], Bbox],
    st_cutouts: Dict[int, np.ndarray],
    ip_correspondences=None,
    id_maps=None,
    corresponding_only=False,
    pdf_writer=None,
    common_scale: bool = False,
    proj_axis: int = 0,
    split_img_loader: SplitImageLoader = None,
):  # pragma: no cover
    """Create a grid plot of subtiles on the tile1-tile2 boundary IP density and include the transformed images.

    Below this function, all pairs are in relation to the tile1-tile2 boundary in this call.

    Parameters
    ----------
    st_overlaps: Dict[Tuple[int, int], Bbox]
        The overlap bounding boxes of the subtiles in this outer boundary.

    st_cutouts: Dict[int, np.ndarray]
        The cutouts of the subtiles in this outer boundary.
    """
    title_mode = "all"

    if proj_axis is None:
        raise NotImplementedError("Split tiles combined plots not implemented for all projections")
    subtiles1, subtiles2 = split_img_loader.get_outer_boundary_subtiles(
        outer_tile1,
        outer_tile2,
        proj_axis=proj_axis,
    )
    fig = plt.figure(figsize=(13, 6))
    outer_grid = fig.add_gridspec(1, 3, wspace=0.2, hspace=0, left=0.08, right=0.92)
    inner_grids = []
    all_axs = []
    for i in range(3):
        inner_grids.append(
            outer_grid[0, i].subgridspec(subtiles1.shape[1], subtiles1.shape[0], wspace=0.1, hspace=0.1)
        )
        all_axs.append(inner_grids[i].subplots(sharex="col", sharey="row"))
    # Determine the interestpoints and mips images per
    # sub tile and their vmin, vmax-es for the left and right panels on a common_scale
    st_pairs = [(int(st1), int(st2)) for (st1, st2) in np.nditer([subtiles1, subtiles2], order='F')]
    st_ips = get_subtile_overlapping_IPs(st_pairs, ip_arrays, tile_transformations, tile_inv_transformations, tile_sizes,
                                         ip_correspondences, id_maps, corresponding_only)

    # Histograms always have the same scale on the left and right
    st_hist_vmax1, st_hist_vmax2 = get_histograms_vmin_vmax(
        st_pairs, st_ips, st_overlaps, proj_axis, common_scale=True
    )
    # Images may have different scales on the left and right
    st_mips, img_vmin1, img_vmax1, img_vmin2, img_vmax2 = get_subtile_mips_and_values(st_pairs,
                                                                                      st_cutouts, common_scale, proj_axis)

    # i_py panel grid index for y, i_px panel grid index for x
    first_panel = True
    for i_py in range(subtiles1.shape[1]):
        for i_px in range(subtiles1.shape[0]):
            # Sub-tile ids
            st1 = subtiles1[i_px, i_py]
            st2 = subtiles2[i_px, i_py]
            # The same sub panel on all three plotting panels
            axs = [all_axs[0][i_py, i_px], all_axs[1][i_py, i_px], all_axs[2][i_py, i_px]]

            if (st1, st2) in st_overlaps:
                w_box_overlap = st_overlaps[(st1, st2)]
                if corresponding_only:
                    title_mode = "corresp."

                cbar_mappable = plot_one_panel_trio(
                    outer_tile1,
                    outer_tile2,
                    st_ips[(st1, st2)][0],
                    st_ips[(st1, st2)][1],
                    st_mips[st1],
                    st_mips[st2],
                    w_box_overlap,
                    proj_axis,
                    axs,
                    fig,
                    img_vmin1,
                    img_vmax1,
                    img_vmin2,
                    img_vmax2,
                    st_hist_vmax1,
                    st_hist_vmax2,
                    common_scale,
                    subtile_plot=True,
                )

                if first_panel:
                    for i in range(3):
                        if cbar_mappable[i]:
                            fig.colorbar(cbar_mappable[i], ax=all_axs[i])
                    first_panel = False

    fig.text(0.01, 0.98, str(subtiles1.T), fontsize=12, ha="left", va="top")
    fig.text(0.90, 0.98, str(subtiles2.T), fontsize=12, ha="left", va="top")
    fig.suptitle(f"Tile{outer_tile1}-{outer_tile2} overlap ({title_mode}) in {AXIS_PROJ[proj_axis]} plane")
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
    image_loader = ZarrImageLoader(xmldict["SpimData"]["SequenceDescription"]["ImageLoader"])
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
                t1,
                t2,
                4,
                tile_transformations,
                tile_inv_transformations,
                tile_full_sizes,
                image_loader=image_loader,
            )
            create_pair_overplot(
                t1, t2, t1_cutout, t2_cutout, w_box_overlap, common_scale=True, pdf_writer=pdf_writer
            )
    with PdfPages(f"{prefix}cutouts_horizontal_overlaps.pdf") as pdf_writer:
        for t1, t2 in horizontal_pairs:
            t1_cutout, t2_cutout, w_box_overlap = get_transformed_pair_cutouts(
                t1,
                t2,
                4,
                tile_transformations,
                tile_inv_transformations,
                tile_full_sizes,
                image_loader=image_loader,
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
    image_loader = ZarrImageLoader(xmldict["SpimData"]["SequenceDescription"]["ImageLoader"])
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
                t1,
                t2,
                4,
                tile_transformations,
                tile_inv_transformations,
                tile_full_sizes,
                image_loader=image_loader,
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
                t1,
                t2,
                4,
                tile_transformations,
                tile_inv_transformations,
                tile_full_sizes,
                image_loader=image_loader,
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


def run_split_combined_plots(
    input_xml: str,
    split_xyz: Tuple[int, int, int],
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
    # get the split image loader
    split_img_loader = SplitImageLoader(xmldict["SpimData"]["SequenceDescription"]["ImageLoader"])
    grid_map = split_img_loader.init_grid_map(split_xyz)
    # get the interest points
    nTiles = split_xyz[0] * split_xyz[1] * split_xyz[2] * 15
    ip_arrays = {}
    ip_correspondences = {}
    id_maps = {}
    # ip_arrays = read_tiles_interestpoints(setup_ids=range(nTiles))
    # ip_correspondences, id_maps = read_ip_correspondences(setup_ids=range(nTiles))
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
    horizontal_pairs = [
        (0, 1), (1, 2), (3, 4), (4, 5), (6, 7), 
        (7, 8), 
        (9, 10), (10, 11), (12, 13), (13, 14)
    ]
    if not prefix:
        prefix = ""
    if vert_proj_axis is None:
        pdf_proj = ""
    else:
        pdf_proj = "_" + AXIS_PROJ[vert_proj_axis]
    with PdfPages(f"{prefix}cutouts_split_vertical_overlaps{pdf_proj}.pdf") as pdf_writer:
        st_cutouts = OrderedDict()
        st_overlaps = OrderedDict()

        for t1, t2 in vertical_pairs:
            LOGGER.info(f"Start processing outer tile pair {t1}-{t2} for vertical overlaps")
            subtiles1, subtiles2 = split_img_loader.get_outer_boundary_subtiles(t1, t2, proj_axis=0)
            for st1, st2 in np.nditer([subtiles1, subtiles2], order="F"):
                st1 = int(st1)
                st2 = int(st2)
                st1_cutout, st2_cutout, w_box_overlap = get_transformed_pair_cutouts(
                    st1,
                    st2,
                    4,
                    tile_transformations,
                    tile_inv_transformations,
                    tile_full_sizes,
                    image_loader=split_img_loader,
                )
                if w_box_overlap is None:
                    LOGGER.warning(f"Tile {t1}-{st1} and {t2}-{st2} has no world overlap boxes. Skipping.")
                    continue
                st_cutouts[st1] = st1_cutout
                st_cutouts[st2] = st2_cutout
                st_overlaps[(st1, st2)] = w_box_overlap


            create_one_projection_split_tiles_figure(
                t1,
                t2,
                ip_arrays,
                tile_transformations,
                tile_inv_transformations,
                tile_full_sizes,
                st_overlaps,
                st_cutouts,
                ip_correspondences,
                id_maps,
                corresponding_only=False,
                pdf_writer=pdf_writer,
                common_scale=True,
                proj_axis=vert_proj_axis,
                split_img_loader=split_img_loader,
            )
            # create_one_projection_combined_figure(
            #     t1,
            #     t2,
            #     ip_arrays,
            #     tile_transformations,
            #     tile_inv_transformations,
            #     tile_full_sizes,
            #     w_box_overlap,
            #     st1_cutout,
            #     st2_cutout,
            #     ip_correspondences,
            #     id_maps,
            #     corresponding_only=True,
            #     pdf_writer=pdf_writer,
            #     common_scale=False,
            #     proj_axis=vert_proj_axis,
            # )
    if hor_proj_axis is None:
        pdf_proj = ""
    else:
        pdf_proj = "_" + AXIS_PROJ[hor_proj_axis]
    with PdfPages(f"{prefix}cutouts_split_horizontal_overlaps{pdf_proj}.pdf") as pdf_writer:
        st_cutouts = OrderedDict()
        st_overlaps = OrderedDict()
        for t1, t2 in horizontal_pairs:
            LOGGER.info(f"Start processing outer tile pair {t1}-{t2} for horizontal overlaps")
            subtiles1, subtiles2 = split_img_loader.get_outer_boundary_subtiles(t1, t2, proj_axis=1)
            for st1, st2 in np.nditer([subtiles1, subtiles2], order="F"):
                st1 = int(st1)
                st2 = int(st2)
                st1_cutout, st2_cutout, w_box_overlap = get_transformed_pair_cutouts(
                    st1,
                    st2,
                    4,
                    tile_transformations,
                    tile_inv_transformations,
                    tile_full_sizes,
                    image_loader=split_img_loader,
                )
                if w_box_overlap is None:
                    LOGGER.warning(f"Tile {t1}-{st1} and {t2}-{st2} has no world overlap boxes. Skipping.")
                    continue
                st_cutouts[st1] = st1_cutout
                st_cutouts[st2] = st2_cutout
                st_overlaps[(st1, st2)] = w_box_overlap

            create_one_projection_split_tiles_figure(
                t1,
                t2,
                ip_arrays,
                tile_transformations,
                tile_inv_transformations,
                tile_full_sizes,
                st_overlaps,
                st_cutouts,
                ip_correspondences,
                id_maps,
                corresponding_only=False,
                pdf_writer=pdf_writer,
                common_scale=True,
                proj_axis=hor_proj_axis,
                split_img_loader=split_img_loader,
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


def run_aff_split_yz_xz_combined_plot():  # pragma: no cover
    """Entry point for run_aff_yz_xz_combined_plot."""
    rlogger = logging.getLogger()
    rlogger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    rlogger.addHandler(handler)
    run_split_combined_plots(
        "../results/bigstitcher.xml",
        split_xyz=(2, 2, 6),
        prefix="../results/aff_",
        vert_proj_axis=0,
        hor_proj_axis=1,
    )

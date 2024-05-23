"""Interestpoint density quality control module."""

from __future__ import annotations

from typing import Optional

import zarr
import numpy as np
import numpy.lib.recfunctions
import json
import xml.etree.ElementTree as ET

from matplotlib.backends.backend_pdf import PdfPages

from .affine_transformation import AffineTransformation
from .bbox import Bbox
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

PROJ_AXIS = {"xy": 2, "xz": 1, "yz": 0}  # pragma: no cover
AXIS_PROJ = {2: "xy", 1: "xz", 0: "yz"}  # pragma: no cover
PROJ_KEEP = {2: np.array([0, 1]), 1: np.array([0, 2]), 0: np.array([1, 2])}  # pragma: no cover


def format_large_numbers(x, pos) -> str:  # pragma: no cover
    """Format large numbers with M and k suffixes.

    pos is not used but required by the matplotlib.ticker.FuncFormatter interface.
    """
    if x >= 1e6 or x <= -1e6:
        return f"{x / 1e6:.0f}M"
    elif x >= 1e3 or x <= -1e3:
        return f"{x / 1e3:.0f}k"
    else:
        return f"{x:.0f}"


def read_json(json_path: str) -> dict:  # pragma: no cover
    """Read a json file and return a dict."""
    with open(json_path) as f:
        return json.load(f)


def read_tiles_interestpoints(
    ip_label="beads", path: str = "../results/interestpoints.n5"
) -> dict[int, np.ndarray]:  # pragma: no cover
    """

    Keeps the xml file axes order, i.e. loc is (x,y,z).

    Parameters
    ----------
    ip_label: str
      Identifier for the interest points. Usually "beads".

    Returns
    -------
    ip_arrays : dict[int, np.ndarray]
        Dictionary of interest point arrays for each tile. At the moment the return dictionary can only
        contain interestpoints of one ip_label.
    """
    n5s = zarr.n5.N5FSStore(path)
    zg = zarr.open(store=n5s, mode="r")
    ip_arrays = {}
    for x in range(15):
        id = zg[f"tpId_0_viewSetupId_{x}/{ip_label}/interestpoints/id"]  # technically 2D
        loc = zg[f"tpId_0_viewSetupId_{x}/{ip_label}/interestpoints/loc"]
        intensities = zg[f"tpId_0_viewSetupId_{x}/{ip_label}/interestpoints/intensities"]

        # Record array with integer id and 3 float column for loc
        T = np.zeros(id.shape[0], dtype=[("id", int), ("loc", float, 3), ("intensity", np.float32)])
        T["id"] = id[:, 0]
        T["loc"] = loc
        T["intensity"] = intensities[:, 0]
        ip_arrays[x] = T
        print(f"Loaded {len(T)} interestpoint for tile {x}")
    return ip_arrays


def read_ip_correspondences(
    ip_label: str = "beads", path: str = "../results/interestpoints.n5"
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, int]]]:  # pragma: no cover
    """Read in the interest point correspondences from the n5 binary files.

    The correspondences are in format: self_id, other_id, map_id

    Parameters
    ----------
    ip_label : str
        Label of IPs that we loaded and looking for correspondences.
        Cross label matching is not supported at the moment.
    """
    n5s = zarr.n5.N5FSStore(path)
    zg = zarr.open(store=n5s, mode="r")
    ip_correspondences = {}
    id_maps = {}
    for x in range(15):
        attr = read_json(f"{path}/tpId_0_viewSetupId_{x}/{ip_label}/correspondences/attributes.json")
        id_maps[x] = attr["idMap"]
        try:
            T = np.array(zg[f"tpId_0_viewSetupId_{x}/{ip_label}/correspondences/data"], dtype=int)
            ip_correspondences[x] = T
            print(f"Loaded {len(T)} interestpoint correspondences for tile {x}")
        except ValueError:
            print(f"No correspondence table for tile {x}")
    return ip_correspondences, id_maps


def get_tile_corresponding_IPs(
    t1: int,
    t2: int,
    ip_arrays: dict[int, np.ndarray],
    ip_correspondences: dict,
    id_maps: dict[int, dict[str, int]],
    ip_label: str = "beads",
):  # pragma: no cover
    """Filter t1's interestpoints that have a correspondence in t2.

    Parameters
    ----------
    ip_arrays : dict[int, np.ndarray]
        Dictionary of interest point arrays for each tile.
    ip_correspondences : dict[int, np.ndarray]
        Dictionary of interest point correspondence array for each tile.
    id_maps : dict[int, dict]
        Dictionary of id maps for each tile.
    R : structured ndarray dtype=[('id', int), ('loc', int, (3,)), ('id2', int)]
    """
    return filter_tile_corresponding_IPs(t1, t2, ip_arrays[t1], ip_correspondences, id_maps, ip_label)


def filter_tile_corresponding_IPs(
    t1: int,
    t2: int,
    t1_ips: np.ndarray,
    ip_correspondences: dict,
    id_maps: dict[int, dict[str, int]],
    t2_ips: np.ndarray = None,
    get_loc2: bool = False,
    ip_label: str = "beads",
):  # pragma: no cover
    """Filter t1's interestpoints that have a correspondence in t2.

    Correspondences are not one-to-one in either direction. Ie. one IP in t1 can
    correspond to multiple IPs in t2 and vice versa. The join operation lists all
    combinations.

    As such neither id nor id2 may be unique in the output.

    Parameters
    ----------
    t1_ips : np.ndarray
        Structured array of interest points in tile 1, with 'id' field.
    ip_correspondences : dict[int, np.ndarray]
        Dictionary of interest point correspondence array for each tile.
    id_maps : dict[int, dict]
        Dictionary of id maps for each tile.
    t2_ips : np.ndarray
        Structured array of interest points in tile 2, with 'id' field. Used only if get_loc2 is True.
    get_loc2 : bool
        If True, the loc field of t2_ips is added as loc2 field to the output array.
    ip_label : str
        IP label used in id_maps of the n5.

    Returns
    -------
    R : np.ndarray
        Structured ndarray with id and id2 fields and keeping all the other already there in the input.
        Returns empty structured array if ip_correspondences for t1 is not found or there is no map
        entry for t1-t2.
    """
    try:
        A = ip_correspondences[t1]
        t2_cid = id_maps[t1][f"0,{t2},{ip_label}"]  # e.g. "0,{t2},beads"
    except KeyError:
        # Create an empty array with the correct dtype fields
        C = np.lib.recfunctions.merge_arrays(
            [t1_ips[np.array([], dtype=int)], np.zeros((0,), dtype=[("id2", int)])], flatten=True
        )
        if get_loc2:
            C = np.lib.recfunctions.drop_fields(C, "id2")
            t2_ips = np.lib.recfunctions.rename_fields(
                t2_ips, {"id": "id2", "loc": "loc2", "intensity": "intensity2"}
            )
            C = np.lib.recfunctions.merge_arrays([C, t2_ips[np.array([], dtype=int)]], flatten=True)
        return C

    A = A[A[:, 2] == t2_cid][:, :2]
    A = np.lib.recfunctions.unstructured_to_structured(A, names=["id", "id2"])
    # np.lib.recfunctions.join_by does not support non unique keys
    pA = pd.DataFrame(A)
    t1_index = np.zeros(t1_ips.shape, dtype=[("id", int), ("index_t1", int)])
    t1_index["id"] = t1_ips["id"]
    t1_index["index_t1"] = np.arange(len(t1_ips))
    pt1_ind = pd.DataFrame(t1_index)
    R = pd.merge(pt1_ind, pA, how="inner", on="id")
    R = R.to_records(index=False)

    C = np.lib.recfunctions.merge_arrays(
        [t1_ips[R["index_t1"]], np.array(R["id2"], dtype=[("id2", int)], copy=False)], flatten=True
    )
    if get_loc2:
        t2_ips = np.lib.recfunctions.rename_fields(
            t2_ips, {"id": "id2", "loc": "loc2", "intensity": "intensity2"}
        )
        t2_index = np.zeros(t2_ips.shape, dtype=[("id2", int), ("index_t2", int)])
        t2_index["id2"] = t2_ips["id2"]
        t2_index["index_t2"] = np.arange(len(t2_ips))
        c_index = np.zeros(C.shape, dtype=[("id2", int), ("index_c", int)])
        c_index["id2"] = C["id2"]
        c_index["index_c"] = np.arange(len(C))
        pt2_ind = pd.DataFrame(t2_index)
        pc_ind = pd.DataFrame(c_index)
        R = pd.merge(pc_ind, pt2_ind, how="inner", on="id2")
        R = R.to_records(index=False)
        C = np.lib.recfunctions.drop_fields(C, "id2")
        C = np.lib.recfunctions.merge_arrays([C[R["index_c"]], t2_ips[R["index_t2"]]], flatten=True)

    return C


def read_tile_transformations(xmlRegistrations: ET.Element) -> dict:  # pragma: no cover
    """Read the tile transformations from the xml file."""
    tile_transformations = {}
    tile_inv_transformations = {}
    for x in range(15):
        xml_reg = xmlRegistrations.find(f"ViewRegistration[@timepoint='0'][@setup='{x}']")
        if xml_reg is None:
            raise ValueError(f"ViewRegistration for setup {x} not found.")
        tile_transformations[x] = AffineTransformation.create_from_xml_ViewRegistration(xml_reg)
        tile_inv_transformations[x] = tile_transformations[x].get_inverse()
    return tile_transformations, tile_inv_transformations


def read_tile_sizes(xml_ViewSetups) -> dict[int, np.ndarray]:  # pragma: no cover
    """Iterate over the 15 tile ids and read their sizes from the ViewSetup section of the xml file.

    Returns
    -------
    tile_sizes : dict[int, np.ndarray]
        Dictionary of tile sizes for each tile. Sizes are in the order
        as defined in the xml file, i.e. x,y,z.
    """
    tile_sizes = {}
    for x in range(15):
        for xml_vSetup in xml_ViewSetups.iterfind("ViewSetup"):
            xml_id = xml_vSetup.find("id")
            if xml_id is not None and x == int(xml_id.text):
                break
        else:
            raise KeyError("setupId {} was not found".format(x))

        xml_sizes = xml_vSetup.find("size")
        tile_sizes[x] = np.array([int(x) for x in xml_sizes.text.strip().split()], dtype=int)
    return tile_sizes


def get_tile_pair_overlap(
    t1, t2, tile_transformations, tile_inv_transformations, tile_sizes
):  # pragma: no cover
    """Get the overlap of two tiles in world and tile coordinates.

    To determine the world overlap, we first transform the tile corners to world coordinates
    make a bounding box around them and then intersecting the two boxes.

    The intersection is then transformed back to tile coordinates and bounding
    boxes are created around them to get the overlap tile coordinates.
    """
    tsizes1 = tile_sizes[t1]
    tsizes2 = tile_sizes[t2]

    # w2t: world coords to tile coords
    # t2w: tile coords to world coords
    t2w_1 = tile_transformations[t1]
    t2w_2 = tile_transformations[t2]

    w_box1 = Bbox.create_box(t2w_1.apply_to([[0, 0, 0], tsizes1]))
    w_box2 = Bbox.create_box(t2w_2.apply_to([[0, 0, 0], tsizes2]))
    try:
        w_box_overlap = w_box1.intersection(w_box2)
    except ValueError:
        return None, None, None  # No overlap
    w2t_1 = tile_inv_transformations[t1]
    w2t_2 = tile_inv_transformations[t2]
    t_box_overlap1 = np.array(
        np.array(w2t_1.apply_to(w_box_overlap.getcorners()), dtype=float) + 0.5, dtype=int
    )
    t_box_overlap1 = np.maximum([[0, 0, 0]], t_box_overlap1)
    t_box_overlap1 = np.minimum(t_box_overlap1, tsizes1)
    t_box_overlap2 = np.array(
        np.array(w2t_2.apply_to(w_box_overlap.getcorners()), dtype=float) + 0.5, dtype=int
    )
    t_box_overlap2 = np.maximum([[0, 0, 0]], t_box_overlap2)
    t_box_overlap2 = np.minimum(t_box_overlap2, tsizes2)
    return w_box_overlap, t_box_overlap1, t_box_overlap2


def get_tile_overlapping_IPs(
    t1, t2, ip1_array, tile_transformations, tile_inv_transformations, tile_sizes
):  # pragma: no cover
    """Get those interestpoints from t1 that are in t2 according to the current affine transformations.

    This method is exact, transforms the interest points to world coordinates and then back to the other tile.

    Parameters
    ----------
    t1 : int
        Tile 1 id
    t2 : int
        Tile 2 id
    ip1_array : np.ndarray
        Structured array of interest points in tile 1, with 'loc' field.

    Returns
    -------
    ip1 : np.ndarray
      Subset of ip1_array that is in t2. This remains a structured array and contains the added field 'loc_w'
      with the world coordinates.
    """
    t2w_1 = tile_transformations[t1]
    w2t_2 = tile_inv_transformations[t2]
    box2 = Bbox.create_box([[0, 0, 0], tile_sizes[t2]])
    ip1 = np.copy(ip1_array)

    newloc = np.zeros(ip1.shape, dtype=[("loc_w", float, 3)])
    ip1 = np.lib.recfunctions.merge_arrays([ip1, newloc], flatten=True)
    # ip1 = np.lib.recfunctions.append_fields(ip1, 'loc_w', np.atleast_2d([0,0,0]), dtypes=[(float, 3)])
    ip1["loc_w"] = t2w_1.apply_to(ip1_array["loc"])
    ip_t1_in_t2 = w2t_2.apply_to(ip1["loc_w"])  # Tile1's IPs in tile 2 coordinates
    mask = box2.contains(ip_t1_in_t2)
    return ip1[mask]


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
        print(f"Tile {tile1} and tile {tile2} do not overlap.")
        return
    print(f"Tile {tile1} and tile {tile2} overlap in world coordinates: {w_box_overlap}")
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
    # for each tile pair
    xml_root = ET.parse(input_xml).getroot()
    xml_ViewSetups = xml_root.find("SequenceDescription/ViewSetups")
    # xml_ImageLoader = xml_root.find("SequenceDescription/ImageLoader")
    xml_ViewRegistrations = xml_root.find("ViewRegistrations")
    # get all transformations
    ip_arrays = read_tiles_interestpoints()
    ip_correspondences, id_maps = read_ip_correspondences()
    tile_sizes = read_tile_sizes(xml_ViewSetups)
    tile_transformations, tile_inv_transformations = read_tile_transformations(xml_ViewRegistrations)
    vertical_pairs = [(0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (9, 12), (10, 13), (11, 14)]
    horizontal_pairs = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (9, 10), (10, 11), (12, 13), (13, 14)]

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
    run_density_plot("../results/bigstitcher.xml", prefix="../results/tr_")


def run_aff_density_plot():  # pragma: no cover
    """Entry point for run_aff_density_plot."""
    run_density_plot("../results/bigstitcher.xml", prefix="../results/aff_")

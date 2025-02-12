"""Tile transformations and overlap functions for QC plotting."""
from __future__ import annotations

import logging
import json
import numpy as np
import numpy.lib.recfunctions
import scipy
import zarr
from collections import OrderedDict
import pandas as pd

from .affine_transformation import AffineTransformation  # pragma: no cover
from .bbox import Bbox  # pragma: no cover

LOGGER = logging.getLogger("tile_transforms")  # pragma: no cover


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


def get_tile_zarr_image_path(tileId: int, xml_dict: OrderedDict) -> str:  # pragma: no cover
    """Get the image's full s3 path from the xml.

    Does not handle properly all relative and absolute base paths.

    Parameters
    ----------
    tileId: int
         The tile id to get the path for
    """
    # Read in the xml as a dict
    img_loader = xml_dict["SpimData"]["SequenceDescription"]["ImageLoader"]
    if "s3bucket" in img_loader:
        prefix = "s3://" + img_loader["s3bucket"].rstrip("/")
    else:
        prefix = xml_dict["SpimData"]["BasePath"]["#text"].rstrip("/")

    zpath = img_loader["zarr"]["#text"].strip("/")
    zpath = prefix + "/" + zpath

    for zgroup in img_loader["zgroups"]["zgroup"]:
        if int(zgroup["@setup"]) == tileId:
            zpath = zpath + "/" + zgroup["path"].strip("/")
            break
    else:
        raise ValueError(f"Tile {tileId} not found in the xml")

    return zpath


def get_tile_slice(
    zgpath: str, level: int, xyz_slices: tuple[slice, slice, slice]
) -> np.ndarray:  # pragma: no cover
    """Return the x,y,z array cutout of the level downsampled version of the image.

    Initiates the loading of the given slice from the zarr array and returns as an ndarray.

    The returned array has axis order of x,y,z.

    Parameters
    ----------
    zgpath: str
        The path to the zarr group containing the image data
    level: int
        The downsampling level of the image pyramid to read from (0,1,2,3,4)
    xyz_slices: tuple[slice, slice, slice]
        The x,y,z slices to read from the image, at the given level.
    """
    # Read in the zarr and get the slice
    LOGGER.info(f"Reading in {zgpath} slices {xyz_slices}")
    z = zarr.open_group(zgpath, mode="r")
    tczyx_slice = (
        0,
        0,
    ) + xyz_slices[
        ::-1
    ]  # The zarr array has axis order of t,c,z,y,x
    return np.array(z[f"{level}"][tczyx_slice]).transpose()


def get_tile_xyz_size(zgpath: str, level: int):  # pragma: no cover
    """Return the x,y,z size of the level downsampled version of the image.

    Parameters
    ----------
    zgpath: str
        The path to the zarr group containing the image data
    level: int
        The downsampling level of the image pyramid to read from (0,1,2,3,4)
    """
    # Read in the zarr and get the size
    z = zarr.open_group(zgpath, mode="r")
    return z[f"{level}"].shape[-3:][::-1]


def read_tile_sizes(xml_ViewSetups: OrderedDict) -> dict[int, np.ndarray]:  # pragma: no cover
    """Read all the tile sizes defined in the given xml section.

    Parameters
    ----------
    xml_ViewSetups : OrderedDict
        The "<ViewSetups>" section of the xml file.

    Returns
    -------
    tile_sizes : dict[int, np.ndarray]
        Dictionary of tile sizes for each tile. Sizes are in the order
        as defined in the xml file, i.e. x,y,z.
    """
    tile_sizes = {}
    vsl = xml_ViewSetups["ViewSetup"]
    if not isinstance(vsl, list):
        # One entry case
        vsl = [
            vsl,
        ]
    for xml_vSetup in vsl:
        x = int(xml_vSetup["id"])
        xml_sizes = xml_vSetup["size"]
        tile_sizes[x] = np.array([int(y) for y in xml_sizes.strip().split()], dtype=int)
    return tile_sizes


def read_tile_transformations(xmlRegistrations: OrderedDict) -> dict:  # pragma: no cover
    """Read the tile transformations from the xml file.

    Does not check whether all tile has a transformation.
    Missing entries will be missing from the return dict.

    Parameters
    ----------
    xmlRegistrations : OrderedDict
        The "<ViewRegistrations>" section of the xml file as an OrderedDict.

    Returns
    -------
    tile_transformations, tile_inv_transformations : dict[int, AffineTransformation]
        Dictionary of tile transformations and its inverse for each tile.
    """
    tile_transformations = {}
    tile_inv_transformations = {}
    vrl = xmlRegistrations["ViewRegistration"]
    if not isinstance(vrl, list):
        # One entry case
        vrl = [vrl, ]
    for xml_reg in vrl:
        x = int(xml_reg["@setup"])
        tp = int(xml_reg["@timepoint"])
        if tp != 0:
            raise ValueError("Unexpected tile layout: Timepoint is not 0")
        tile_transformations[x] = AffineTransformation.create_from_xmldict_ViewRegistration(xml_reg)
        tile_inv_transformations[x] = tile_transformations[x].get_inverse()
    return tile_transformations, tile_inv_transformations


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

    t1box = Bbox.create_box([[0, 0, 0], tsizes1])
    t2box = Bbox.create_box([[0, 0, 0], tsizes2])
    # All corners should be considered as the transformation may contain flips or large rotations
    w_box1 = Bbox.create_box(t2w_1.apply_to(t1box.getallcorners()))
    w_box2 = Bbox.create_box(t2w_2.apply_to(t2box.getallcorners()))
    try:
        w_box_overlap = w_box1.intersection(w_box2)
    except ValueError:
        return None, None, None  # No overlap
    w2t_1 = tile_inv_transformations[t1]
    w2t_2 = tile_inv_transformations[t2]
    t_box_overlap1 = Bbox.create_box(w2t_1.apply_to(w_box_overlap.getallcorners())).ensure_ints()
    t_box_overlap1 = t_box_overlap1.intersection(t1box)
    t_box_overlap2 = Bbox.create_box(w2t_2.apply_to(w_box_overlap.getallcorners())).ensure_ints()
    t_box_overlap2 = t_box_overlap2.intersection(t2box)
    LOGGER.info(f"w_box_overlap = {w_box_overlap}")
    return w_box_overlap, t_box_overlap1, t_box_overlap2


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
        LOGGER.info(f"Loaded {len(T)} interestpoint for tile {x}")
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
            LOGGER.info(f"Loaded {len(T)} interestpoint correspondences for tile {x}")
        except ValueError:
            LOGGER.info(f"No correspondence table for tile {x}")
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
    # pd DataFrame does not support fields that are arrays themselves (here the loc has 3 floats)
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


def get_tile_overlapping_IPs(
    t1, t2, ip1_array, tile_transformations, tile_inv_transformations, tile_sizes
) -> np.ndarray:  # pragma: no cover
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
    ip1 : Structured np.ndarray
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


def get_transformed_tile_cutout(
    tileId: int, w_box_overlap: Bbox, level: int, tile_inv_transformations: dict, xmldict: OrderedDict
):  # pragma: no cover
    """Get the transformed cutout of the tile for a given world coordinate box.

    This function transforms the read tile cutout into the world coordinate box
    by `scipy.ndimage.affine_transform`.

    Parameters
    ----------
    tileId : int
        The tile id to get the cutout for
    w_box_overlap : Bbox
        The overlap box in world coordinates
    level : int
        The downsampling level of the image pyramid to read from (0,1,2,3,4)

    Returns
    -------
    target_tile1 : np.ndarray
        The transformed cutout of the tile, a 3D array with axis order of x,y,z
    dtarget_box : Bbox
        The bounding box of the transformed cutout in world coordinates
        at the target level i.e. in a downsampled world coordinate system.
    """

    w2t_1 = tile_inv_transformations[tileId]  # World to tile coordinates transformation

    # Upscale from level to level 0 and downscale from level 0 to level
    upscale_t = AffineTransformation.create_upscale_transformation(1 << level)
    downscale_t = upscale_t.get_inverse()
    zp1 = get_tile_zarr_image_path(tileId, xmldict)
    d_tsizes1 = get_tile_xyz_size(zp1, level)  # The image size of the downsampled tile

    t_box_overlap1 = Bbox.create_box(w2t_1.apply_to(w_box_overlap.getallcorners()))
    # These are coordinates in the downscaled tiles - may be out of bounds but that's ok
    # The affine transform resampling will recognize this and fill with zeros
    # Downscaling/upscaling keeps the bottom-left, top-right corners of the box
    ds_t_box_overlap1 = downscale_t.apply_to(t_box_overlap1.getcorners()).astype(int)
    # Indices should not go out of bounds (0,0,0), (xsize,ysize,zsize)
    ds_t_box_overlap1 = np.maximum([[0, 0, 0]], ds_t_box_overlap1)
    ds_t_box_overlap1 = np.minimum(ds_t_box_overlap1, d_tsizes1)
    ds_t_box_overlap1 = Bbox.create_box(ds_t_box_overlap1)  # This is the image box that we will load

    target_corners = downscale_t.apply_to(w_box_overlap.getcorners())
    LOGGER.info(f"target_corners = {target_corners}")
    # The target box in downsampled coordinates, with a downsampled world coordinate for bottom left corner
    dtarget_box = Bbox.create_box(target_corners)
    dtarget_box.ensure_ints()
    d_target_shape = dtarget_box.getsizes()  # The size of the target image we want to create
    # Constructing the transformation from the target image coordinates to the source image coordinates
    T = AffineTransformation(translation=dtarget_box.get_from_origin_translation())
    T.left_compose(upscale_t)
    T.left_compose(w2t_1)
    T.left_compose(downscale_t)
    # The reading "moves" the image to 0,0,0
    T.left_compose(AffineTransformation(translation=ds_t_box_overlap1.get_to_origin_translation()))

    LOGGER.info("Destination cutout in original image: {} ".format(T.apply_to([[0, 0, 0], d_target_shape])))
    arr_tile1_cutout = get_tile_slice(zp1, level, ds_t_box_overlap1.getslices())
    LOGGER.info(f"Transforming image of size {ds_t_box_overlap1.getsizes()} ")
    target_tile1 = scipy.ndimage.affine_transform(
        arr_tile1_cutout, T.hctransform, output_shape=d_target_shape, order=1
    )
    LOGGER.info("Transformation done")
    return target_tile1, dtarget_box


def get_transformed_pair_cutouts(
    t1: int,
    t2: int,
    level: int,
    tile_transformations: dict,
    tile_inv_transformations: dict,
    tile_sizes: dict,
    xmldict: OrderedDict,
):  # pragma: no cover
    """Get the cutouts of the two tiles in their overlapping region.

    Determine the overlap world bbox of the two tiles as the intersection of the
    world coordinate enclosing bbox-es of the individual tiles.
    """
    # The overlap box in world coordinates, at full scale
    w_box_overlap, _, _ = get_tile_pair_overlap(
        t1, t2, tile_transformations, tile_inv_transformations, tile_sizes
    )
    # get overlap box
    if w_box_overlap is None:
        return None, None, None
    # get transformed cutouts
    t1_cutout, t1_box = get_transformed_tile_cutout(
        t1, w_box_overlap, level, tile_inv_transformations, xmldict
    )
    t2_cutout, t2_box = get_transformed_tile_cutout(
        t2, w_box_overlap, level, tile_inv_transformations, xmldict
    )
    return t1_cutout, t2_cutout, w_box_overlap

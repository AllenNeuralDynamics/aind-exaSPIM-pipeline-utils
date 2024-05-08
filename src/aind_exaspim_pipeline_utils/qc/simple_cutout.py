"""Create transformed output images from input images by scipy.ndimage.affine_transform."""
from collections import OrderedDict
from typing import Optional

import xml.etree.ElementTree as ET
import numpy as np
import scipy.ndimage
import xmltodict
import zarr
from .bbox import Bbox
from .affine_transformation import AffineTransformation


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


def get_tile_slice(zgpath: str, level: int, xyz_slices: tuple[slice, slice, slice]):  # pragma: no cover
    """Return the x,y,z array cutout of the level downsampled version of the image.

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
    z = zarr.open_group(zgpath, mode="r")
    tczyx_slice = (
        0,
        0,
    ) + xyz_slices[::-1]
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


def read_tile_sizes(xml_ViewSetups):  # pragma: no cover
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


def run_cutout_plot(input_xml: str, prefix: Optional[str] = None):  # pragma: no cover
    """Extract overlap from tile pairs

    interestpoints.n5 must be in the current working directory.
    """
    xml_root = ET.parse(input_xml).getroot()
    xml_ViewSetups = xml_root.find("SequenceDescription/ViewSetups")
    xml_ViewRegistrations = xml_root.find("ViewRegistrations")
    with open(input_xml) as f:
        xmldict = xmltodict.parse(f.read())
    # get all transformations
    tile_full_sizes = read_tile_sizes(xml_ViewSetups)
    tile_transformations, tile_inv_transformations = read_tile_transformations(xml_ViewRegistrations)

    t1 = 4
    t2 = 7
    level = 4
    # deteremine the overlap box in world coordinates
    tsizes1 = tile_full_sizes[t1]
    tsizes2 = tile_full_sizes[t2]

    # w2t: world coords to tile coords
    # t2w: tile coords to world coords
    t2w_1 = tile_transformations[t1]
    t2w_2 = tile_transformations[t2]

    w_box1 = Bbox.create_box(t2w_1.apply_to(Bbox.create_box([[0, 0, 0], tsizes1]).getallcorners()))
    w_box2 = Bbox.create_box(t2w_2.apply_to(Bbox.create_box([[0, 0, 0], tsizes2]).getallcorners()))
    try:
        w_box_overlap = w_box1.intersection(w_box2)
    except ValueError:
        raise RuntimeError(f"Tile {t1} and Tile {t2} do not overlap.")
    w2t_1 = tile_inv_transformations[t1]
    # w2t_2 = tile_inv_transformations[t2]

    upscale_t = AffineTransformation.create_upscale_transformation(1 << level)
    downscale_t = upscale_t.get_inverse()
    zp1 = get_tile_zarr_image_path(t1, xmldict)
    # zp2 = get_tile_zarr_image_path(t2, xmldict)
    d_tsizes1 = get_tile_xyz_size(zp1, level)

    t_box_overlap1 = Bbox.create_box(w2t_1.apply_to(w_box_overlap.getallcorners()))
    # t_box_overlap1.ensure_ints()
    # t_box_overlap1 = np.maximum([[0, 0, 0]], t_box_overlap1)
    # t_box_overlap1 = np.minimum(t_box_overlap1, tsizes1)
    # These are coordinates in the downscaled tiles - may be out of bounds but that's ok
    # The affine transform resampling will recognize this and fill with zeros
    ds_t_box_overlap1 = downscale_t.apply_to(t_box_overlap1.getcorners()).astype(int)
    ds_t_box_overlap1 = np.maximum([[0, 0, 0]], ds_t_box_overlap1)
    ds_t_box_overlap1 = np.minimum(ds_t_box_overlap1, d_tsizes1)

    target_corners = downscale_t.apply_to(w_box_overlap.getcorners())
    # The target box in downsampled coordinates, with a downsampled world coordinate for bottom left corner
    dtarget_box = Bbox.create_box(target_corners)
    dtarget_box.ensure_ints()
    d_target_shape = dtarget_box.getsizes()  # The size of the target image we want to create
    # Constructing the transformation from the target image coordinates to the source image coordinates
    T = AffineTransformation(translation=dtarget_box.get_from_origin_translation())
    T.left_compose(upscale_t)
    T.left_compose(w2t_1)
    T.left_compose(downscale_t)

    arr_tile1_cutout = get_tile_slice(zp1, level, Bbox.create_box(ds_t_box_overlap1).getslices())

    target_tile1 = scipy.ndimage.affine_transform(
        arr_tile1_cutout, T.hctransform, output_shape=d_target_shape
    )
    return target_tile1, dtarget_box, w_box_overlap

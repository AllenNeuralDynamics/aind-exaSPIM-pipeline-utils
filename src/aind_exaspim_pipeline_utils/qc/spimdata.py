"""Classes to read in SpimData XML files and extract information from them including the split-tile case."""
import copy
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Tuple, List, Optional

import numpy as np
import zarr

LOGGER = logging.getLogger("spimdata")


class ImageLoaderABC(ABC):
    @abstractmethod
    def get_tile_xyz_size(self, tileId: int, level: int):
        pass

    @abstractmethod
    def get_tile_slice(self, tileId: int, level: int, xyz_slices: tuple[slice, slice, slice]) -> np.ndarray:
        pass


class ZarrImageLoader(ImageLoaderABC):
    """The base class to read-in ImageLoader section of the SpimData XML file."""

    def __init__(self, xmlImageLoader: OrderedDict, basePath: str = None):
        """Initialize the ImageLoader object.

        Makes an internal copy of the xmlDict.

        Parameters
        ----------
        xmlImageLoader: OrderedDict
            The xml file <ImageLoader> section as an OrderedDict.
        """
        self.xmlImageLoader = copy.deepcopy(xmlImageLoader)
        self.validate_imageloader_format()
        if basePath is None:
            self.basePath = ""
        else:
            self.basePath = basePath.rstrip("/")
        self.tile_zarr_paths: Dict[int, str] = self.init_tile_paths()

    def validate_imageloader_format(self) -> None:
        """Validate the ImageLoader section of the xml."""
        if self.xmlImageLoader["@format"] != "bdv.multimg.zarr":
            raise ValueError("This class is for zarr image loading only.")

    def init_tile_paths(self) -> Dict[int, str]:
        """Initialize the internal tile path dictionary."""
        zpath = self.xmlImageLoader["zarr"]["#text"].strip("/")
        tile_paths = {}
        if "s3bucket" in self.xmlImageLoader:
            zpath = "s3://" + self.xmlImageLoader["s3bucket"].rstrip("/") + "/" + zpath
        elif self.xmlImageLoader["zarr"]["@type"] == "relative":
            zpath = self.basePath + "/" + zpath

        zgl = self.xmlImageLoader["zgroups"]["zgroup"]
        if not isinstance(zgl, list):
            zgl = [
                zgl,
            ]
        for zgroup in zgl:
            tileId = int(zgroup["@setup"])
            tile_paths[tileId] = zpath + "/" + zgroup["path"].strip("/")
        return tile_paths

    def get_tile_zarr_image_path(self, tileId: int) -> str:
        """Get the image's full s3 path from the xml.

        Does not handle properly all relative and absolute base path scenarios.

        Parameters
        ----------
        tileId: int
             The tile id to get the path for
        """
        return self.tile_zarr_paths[tileId]

    def get_tile_xyz_size(self, tileId: int, level: int):  # pragma: no cover
        """Return the x,y,z size of the level downsampled version of the image.

        The value is read from the zarr storage.

        Parameters
        ----------
        zgpath: str
            The path to the zarr group containing the image data
        level: int
            The downsampling level of the image pyramid to read from (0,1,2,3,4)

        Returns
        -------
        sizes: np.ndarray
            The x,y,z size of the image at the given level.
        """
        # Read in the zarr and get the size
        zgpath = self.get_tile_zarr_image_path(tileId)
        z = zarr.open_group(zgpath, mode="r")
        return np.array(z[f"{level}"].shape[-3:][::-1], dtype=int)

    def get_tile_slice(
        self, tileId: int, level: int, xyz_slices: tuple[slice, slice, slice]
    ) -> np.ndarray:  # pragma: no cover
        """Return the x,y,z array cutout of the level downsampled version of the image.

        Initiates the loading of the given slice from the zarr array and returns as an ndarray.

        The returned array has axis order of x,y,z.

        Parameters
        ----------
        tileId: int
            The tile id.
        level: int
            The downsampling level of the image pyramid to read from (0,1,2,3,4)
        xyz_slices: tuple[slice, slice, slice]
            The x,y,z slices to read from the image, at the given level.
        """
        # Read in the zarr and get the slice
        zgpath = self.get_tile_zarr_image_path(tileId)
        LOGGER.info(f"Reading in {zgpath} slices {xyz_slices}")
        z = zarr.open_group(zgpath, mode="r")
        tczyx_slice = (
            0,
            0,
        ) + xyz_slices[
            ::-1
        ]  # The zarr array has axis order of t,c,z,y,x
        return np.array(z[f"{level}"][tczyx_slice]).transpose()


class SplitImageLoader(ImageLoaderABC):
    """The base class to read-in SplitImageLoader section of the SpimData XML file."""

    def __init__(self, xmlSplitImageLoader: OrderedDict, basePath: str = None):
        """Initialize the SplitImageLoader object.

        Makes an internal copy of the xmlDict.

        Parameters
        ----------
        xmlSplitImageLoader: OrderedDict
            The xml file outer <ImageLoader> section as an OrderedDict.
        """
        self.xmlSplitImageLoader = copy.deepcopy(xmlSplitImageLoader)
        self.validate_split_imageloader_format()
        self.basePath = basePath  # Base path may be necessary for the inner loader
        self.innerLoader = ZarrImageLoader(self.xmlSplitImageLoader["ImageLoader"], basePath=basePath)
        # The mapping of the split tile ids to the inner tile ids
        self.tileIdMapping: Dict[int, int] = {}
        self.tileIdReverseMapping: Dict[int, List[int]] = {}
        # The mapping of the new ids to the min and max coordinates in the inner tile
        self.tileSplitMapping: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.tileIdMapping, self.tileIdReverseMapping, self.tileSplitMapping = self.init_tile_mapping()

    def init_tile_mapping(self) -> Tuple[Dict, Dict]:
        """Initialize the internal tile mapping dictionary."""
        tile_split_mapping = OrderedDict()
        tile_id_mapping = OrderedDict()
        tile_reverse_mapping = dict()
        setup_id_defs = self.xmlSplitImageLoader["SetupIds"]["SetupIdDefinition"]
        if not isinstance(setup_id_defs, list):
            setup_id_defs = [
                setup_id_defs,
            ]
        for id_def in setup_id_defs:
            newId = int(id_def["NewId"])
            oldId = int(id_def["OldId"])
            if oldId not in tile_reverse_mapping:
                tile_reverse_mapping[oldId] = []
            tile_reverse_mapping[oldId].append(newId)
            if newId in tile_split_mapping:
                raise ValueError(f"Tile id {newId} is already in the mapping.")
            mincoords = np.array([int(y) for y in id_def["min"].strip().split()], dtype=int)
            maxcoords = np.array([int(y) for y in id_def["max"].strip().split()], dtype=int)
            tile_id_mapping[newId] = oldId
            tile_split_mapping[newId] = (mincoords, maxcoords)
        for x in tile_reverse_mapping:
            tile_reverse_mapping[x] = np.array(tile_reverse_mapping[x])
        return tile_id_mapping, tile_reverse_mapping, tile_split_mapping

    def validate_split_imageloader_format(self) -> None:
        """Validate the SplitImageLoader section of the xml."""
        if self.xmlSplitImageLoader["@format"] != "split.viewerimgloader":
            raise ValueError("This class is for split image loading only.")

    def init_grid_map(self, xyz_grid_size: Tuple[int, int, int]) -> Dict[int, np.ndarray]:
        """Organize the old->new tile ids mapping onto an xyz grid per each old tile id.

        Assume that the newIds are increasing monotonically per OldId, if not, raises ValueError.
        Assume that the newIds are fastest changing by x, then y, then z.

        Parameters
        ----------
        xyz_grid_size: Tuple[int, int, int]
            The size of the sub-tiling grid in x,y,z directions. Each tile must be
            split in the same way.

        Returns
        -------
        grid_map: Dict[int,np.ndarray]
            The grid mapping of the new tile ids per old tile id.
            Each array element has shape xyz_grid, ie. axis order is x,y,z.
        """
        grid_map = {}
        n_subtiles = xyz_grid_size[0] * xyz_grid_size[1] * xyz_grid_size[2]
        for oldId, newIds in self.tileIdReverseMapping.items():
            if len(newIds) != n_subtiles:
                raise ValueError(f"NewIds for oldId {oldId} do not match the xyz_grid_size.")
            if not np.all(np.diff(newIds) == 1):
                raise ValueError(f"NewIds for oldId {oldId} are not sequential.")
            # x is the fastest changing index
            grid_map[oldId] = np.array(newIds).reshape(xyz_grid_size, order="F")
        self.subtileGridMap = grid_map
        return grid_map

    def get_tile_xyz_size(self, tileId: int, level: int):  # pragma: no cover
        """Return the x,y,z size of the level downsampled version of the image.

        The value is read from the zarr storage.

        Parameters
        ----------
        zgpath: str
            The path to the zarr group containing the image data
        level: int
            The downsampling level of the image pyramid to read from (0,1,2,3,4)
        """
        # Read in the zarr and get the size
        xyz_size_oldtile = self.innerLoader.get_tile_xyz_size(self.tileIdMapping[tileId], level)
        factor = 1 << level
        min_offsets, max_offsets = self.tileSplitMapping[tileId]
        dscale_max_offsets = max_offsets // factor
        dscale_min_offsets = min_offsets // factor
        max_offsets = np.minimum(np.array(xyz_size_oldtile), dscale_max_offsets)
        sizes = max_offsets - dscale_min_offsets
        return sizes

    def get_outer_boundary_subtiles(
        self, old_t1: int, old_t2: int, proj_axis: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gets a 2D array of the new tile Ids on the outer boundary of the old tile.
        The old tiles must have an overlap along proj_axis.

        Parameters
        ----------
        old_t1, old_t2: int
            The old tile id of the first and the second tile where old_t1 < old_t2.
        proj_axis: int
            The axis along which the overlap is present, must be 0 == x or 1 == y.

        Returns
        -------
        t1, t2: np.ndarray
            The 2D arrays of the new tile ids on the outer boundary of the old tiles. Should overlap element-wise.
            The 2D arrays have spatial axis order of x,y.
        """
        if old_t1 >= old_t2:
            raise ValueError("old_t1 must be less than old_t2.")
        if proj_axis == 0:
            # x direction overlap
            t1 = self.subtileGridMap[old_t1][0, :, :]
            t2 = self.subtileGridMap[old_t2][-1, :, :]
        elif proj_axis == 1:
            # y direction overlap
            t1 = self.subtileGridMap[old_t1][:, -1, :]
            t2 = self.subtileGridMap[old_t2][:, 0, :]
        else:
            raise ValueError("proj_axis must be 0 or 1 for outer boundary subtiles.")
        LOGGER.info(
            f"Outer boundary subtiles for {old_t1} and {old_t2} in {proj_axis} axis " f"are {t1} and {t2}."
        )
        return t1, t2

    def get_inner_boundary_subtiles(
        self, old_t: int, proj_axis: int, inner_index: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gets a 2D array of the new tile Ids at an inner boundary within an old tile.

        Parameters
        ----------
        old_t: int
            The old tile id where from where we want the inner boundary subtiles.
        proj_axis: int
            The axis along which the overlap is present, 0 == x, 1 == y or 2 == z.
        inner_index: int > 0 or None
            The index of the inner boundary in the proj_axis direction. If None, it is the middle of the axis.
            The overlap is between inner_index-1 and inner_index.

        Returns
        -------
        t1, t2: np.ndarray
            The 2D arrays of the new tile ids at the inner boundary of the old tiles. Should overlap element-wise.
            The 2D arrays have spatial axis order of x,y.
        """
        if inner_index is None:
            inner_index = self.subtileGridMap[old_t].shape[proj_axis] // 2
            if inner_index == 0:
                raise ValueError("Size in the requested axis must be at least 2")

        if proj_axis == 0:
            s1 = (inner_index - 1, slice(None), slice(None))
            s2 = (inner_index, slice(None), slice(None))
        elif proj_axis == 1:
            s1 = (slice(None), inner_index - 1, slice(None))
            s2 = (slice(None), inner_index, slice(None))
        elif proj_axis == 2:
            s1 = (slice(None), slice(None), inner_index - 1)
            s2 = (slice(None), slice(None), inner_index)

        t1 = self.subtileGridMap[old_t][s1]
        t2 = self.subtileGridMap[old_t][s2]
        LOGGER.info(f"Inner boundary subtiles for {old_t} in {proj_axis} axis " f"are {t1} and {t2}.")

        return t1, t2

    def get_tile_slice(
        self, tileId: int, level: int, xyz_slices: tuple[slice, slice, slice]
    ) -> np.ndarray:  # pragma: no cover
        """Return the x,y,z array cutout of the level downsampled version of the image.

        Initiates the loading of the given slice from the zarr array and returns as an ndarray.

        The returned array has axis order of x,y,z.

        Parameters
        ----------
        tileId: int
            The tile id.
        level: int
            The downsampling level of the image pyramid to read from (0,1,2,3,4)
        xyz_slices: tuple[slice, slice, slice]
            The x,y,z slices to read from the image, at the given level.
        """
        # Read in the zarr and get the slice
        factor = 1 << level
        min_offsets, max_offsets = self.tileSplitMapping[tileId]
        inner_slices = []
        for i, s in enumerate(xyz_slices):
            start, stop, step = s.start, s.stop, s.step
            dscale_min_offset = min_offsets[i] // factor
            dscale_max_offset = max_offsets[i] // factor
            if start is not None:
                if start < 0:
                    start += dscale_max_offset
                else:
                    start += dscale_min_offset
            if stop is not None:
                if stop < 0:
                    stop += dscale_max_offset
                else:
                    stop += dscale_min_offset
            inner_slices.append(slice(start, stop, step))

        return self.innerLoader.get_tile_slice(self.tileIdMapping[tileId], level, tuple(inner_slices))

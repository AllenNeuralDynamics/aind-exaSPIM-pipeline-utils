"""Classes to read in SpimData XML files and extract information from them including the split-tile case."""
import copy
import logging
from collections import OrderedDict
from typing import Dict

import numpy as np
import zarr

LOGGER = logging.getLogger("spimdata")

class ZarrImageLoader:
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
            raise ValueError("Only zarr format is supported")

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
            zgl = [self.xmlImageLoader["zgroups"]["zgroup"], ]
        for zgroup in zgl:
            tileId =  int(zgroup["@setup"])
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
        """
        # Read in the zarr and get the size
        zgpath = self.get_tile_zarr_image_path(tileId)
        z = zarr.open_group(zgpath, mode="r")
        return z[f"{level}"].shape[-3:][::-1]

    def get_tile_slice(self,
        tileId: int, level: int, xyz_slices: tuple[slice, slice, slice]
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

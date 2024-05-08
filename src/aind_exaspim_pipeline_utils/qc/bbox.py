"""Simple axis aligned bounding box class."""

from __future__ import annotations  # pragma: no cover
import numpy as np  # pragma: no cover
import numpy.typing as npt  # pragma: no cover


class Bbox:  # pragma: no cover
    """Simple bounding box class"""

    bleft = None
    tright = None

    def __init__(self, bleft, tright):
        """Create a bounding box from bottom left and top right coordinates.

        Raises
        ------
        ValueError
            If the bottom left corner coordinates are not all less than or equal to the top right ones.

        Parameters
        ----------
        bleft, tright : array-like (dim,)
            Bottom left and top right corner coordinates.
        """
        bleft = np.atleast_1d(bleft)
        tright = np.atleast_1d(tright)
        if not np.all(bleft <= tright):
            raise ValueError(
                "Bbox: Bottom left corner coordinates must all be " " less than or equal to top right ones."
            )
        self.bleft = bleft
        self.tright = tright

    def intersection(self, other: Bbox):
        """Return the intersection of two bounding boxes.

        Raises
        ------
        ValueError
            If the two bounding boxes do not intersect.
        """
        bleft = np.maximum(self.bleft, other.bleft)
        tright = np.minimum(self.tright, other.tright)
        return Bbox(bleft, tright)

    def union(self, other: Bbox):
        """Return the union of two bounding boxes."""
        bleft = np.minimum(self.bleft, other.bleft)
        tright = np.maximum(self.tright, other.tright)
        return Bbox(bleft, tright)

    @staticmethod
    def create_box(points):
        """Create a new bounding box from a set of points.

        Parameters
        ----------
        points : array-like (N, dim)
            N points in dim dimensions.
        """
        x = np.atleast_2d(points)
        bleft = np.amin(x, axis=0)
        tright = np.amax(x, axis=0)
        return Bbox(bleft, tright)

    def getcorners(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the bottom left and top right corners of the bounding box."""
        return self.bleft, self.tright

    def getallcorners(self):
        """Returns all corners of the bounding box.

        This works only for 3D boxes.
        """
        corners = np.array(
            [
                [self.bleft[0], self.bleft[1], self.bleft[2]],
                [self.bleft[0], self.bleft[1], self.tright[2]],
                [self.bleft[0], self.tright[1], self.bleft[2]],
                [self.bleft[0], self.tright[1], self.tright[2]],
                [self.tright[0], self.bleft[1], self.bleft[2]],
                [self.tright[0], self.bleft[1], self.tright[2]],
                [self.tright[0], self.tright[1], self.bleft[2]],
                [self.tright[0], self.tright[1], self.tright[2]],
            ]
        )
        return corners

    def getslices(self):
        """Return the box left-right coordinates as slices.

        Must be integer type. tright is interpreted as the not-included limit"""
        return tuple(slice(x, y) for x, y in zip(self.bleft, self.tright))

    def contains(self, points: npt.ArrayLike):
        """Check if a set of points is contained in the bounding box.


        """
        points = np.atleast_2d(points)
        return np.all(points >= self.bleft, axis=1) & np.all(points < self.tright, axis=1)

    def ensure_ints(self):
        """Rounds the bottom left and top right coordinates and casts them to integers in-place.

        For convenience, returns self.
        """
        self.bleft = np.round(self.bleft).astype(int)
        self.tright = np.round(self.tright).astype(int)
        return self

    def getsizes(self) -> np.ndarray:
        """Return the size of the bounding box in each dimension.

        For integer boxes, it can be interpreted as [left, right) limits, i.e. [1,2] has a size of 1.
        """
        return self.tright - self.bleft

    def get_to_origin_translation(self) -> np.ndarray:
        """Returns the translation to move the box to the origin.

        These are -1 * the coordinates of the bottom left corner.
        """
        return -1 * self.bleft

    def get_from_origin_translation(self) -> np.ndarray:
        """Returns the translation to move the box from the origin to its current position.

        These are the coordinates of the bottom left corner.
        """
        return self.bleft

    def move_to_origin(self):
        """Moves the box to the origin by subtracting the bottom left corner
        from the top right one in-place."""
        self.tright -= self.bleft
        self.bleft = np.zeros_like(self.bleft)

    def __str__(self):
        """Return a string representation of the bounding box."""
        return f"Bbox(bleft={self.bleft}, tright={self.tright})"

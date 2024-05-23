"""Affine transformations represented as numpy matrices."""
from __future__ import annotations
import numpy as np
from typing import Optional, Union, Dict, List, Any
import numpy.typing as npt
from collections.abc import Iterable
import xml.etree.ElementTree as ET


class AffineTransformation:
    """Affine transformation representation class.

    Store a homogeneous coordinates representation of an affine transformation.

    We use the notation that linear operators are performed right-to-left.
    """

    def __init__(
        self,
        *,
        scale: Optional[npt.ArrayLike] = None,
        array: Optional[npt.ArrayLike] = None,
        translation: Optional[npt.ArrayLike] = None,
        ndim: int = 3,
    ):
        """Create an affine transformation instance.

        The array (linear operator) part of the affine transformation can be given as a (ndim, ndim) array
        in ``array`` or a scaling vector or a scaling factor in ``scale`. The translation can be given as a
        vector of shape (ndim, ) or a single offset that applies to all dimensions in ``translation``. If
        none of them are specified, initializes an identity transformation.

        Parameters
        ----------
        scale: `numpy.ndarray` broadcastable to (ndim, ).
            Initializes the transformation with these scaling factors. Can be combined with ``translation``.
        array: `numpy.ndarray` broadcastable to (ndim, ndim).
            The array part of the affine transformation. Can be combined with ``translation``.
        translation: `numpy.ndarray` broadcastable to (ndim, ).
            Initializes the transformation with this translation vector. Can be combined with ``scale``.
        ndim: `int` default=3
            Number of dimensions for the transformation.
        """
        self.NDIM = ndim
        self.NDP1 = ndim + 1
        # The affine transformation represented in homogeneous coordinates
        self._hctransform = np.identity(self.NDP1, dtype=float)

        if scale is not None:
            i_diag = np.arange(self.NDIM, dtype=int)
            self._hctransform[i_diag, i_diag] = scale
        if array is not None:
            self._hctransform[: self.NDIM, : self.NDIM] = array
        if translation is not None:
            # The translation is the upper 3 elements in the last column
            self._hctransform[: self.NDIM, self.NDIM] = translation

    def copy(self) -> AffineTransformation:
        """Copy the AffineTransformation instance by copying its internal array."""
        R = AffineTransformation()
        R._hctransform[...] = self._hctransform
        return R

    def get_inverse(self) -> AffineTransformation:
        """Calculate the inverse affine transformation.

        Returns
        -------
        R: `AffineTransformation`
          A new transformation instance that is the inverse of this one.
        """
        R = AffineTransformation()
        R._hctransform = np.linalg.inv(self._hctransform)
        return R

    def get_matrix(self) -> np.ndarray:
        """Get the matrix (linear operator) part of the affine transformation.

        Returns
        -------
        R: `np.ndarray` of `float` (NDIM, NDIM) shape
          Return a view to the matrix part of the affine transformation.
        """
        return self._hctransform[: self.NDIM, : self.NDIM]

    def get_translation(self) -> np.ndarray:
        """Get the translation (vector) part of the affine transformation.

        Returns
        -------
        V: `np.ndarray` of `float` (ndim, ) shape
          Return a view to the translation vector.
        """
        return self._hctransform[: self.NDIM, self.NDIM]

    def add_translation(self, V: npt.ArrayLike) -> None:
        """Add the translation (vector) to the affine transformation.

        Parameters
        ----------
        V: `np.ndarray` broadcastable to (ndim, ) shape
          Add this translation to the transformation.
        """
        self._hctransform[: self.NDIM, self.NDIM] += V

    def apply_to(self, V: npt.ArrayLike) -> np.ndarray:
        """Apply the transformation to a vector or a number of vectors.

        Parameters
        ----------
        V : `numpy.ndarray`
            One vector of shape (NDIM, ) or an arbitrary shape array of multiple vectors (..., NDIM)

        Returns
        -------
        R : `numpy.array` of `float`
           Same shape as ``V``. The affine transformation applied to the vector(s).
        """
        V = np.array(V, copy=False)  # Ensure that this is an array
        if V.shape[-1] == self.NDIM:
            tmp_vec = np.ones(V.shape[:-1] + (self.NDP1, 1), dtype=float)
            tmp_vec[..., : self.NDIM, 0] = V
        else:
            raise ValueError(f"Vector array has wrong shape {V.shape}")
        return np.matmul(self._hctransform, tmp_vec)[..., : self.NDIM, 0]

    def right_compose(self, T: AffineTransformation) -> None:
        """Compose ``self`` with ``T`` in place.

        The resulting transformation is `self composed with T`, i.e.
        it performs T first, then self.

        Parameters
        ----------
        T: `AffineTransformation`
            The transformation to append self to.
        """
        self._hctransform = np.matmul(self._hctransform, T._hctransform)

    def left_compose(self, T: AffineTransformation) -> None:
        """Compose ``T`` with ``self`` in place.

        The resulting transformation is ``T composed with self``, i.e.
        it performs self first, then T.

        Parameters
        ----------
        T: `AffineTransformation`
            Self is appended to this transformation.
        """
        self._hctransform = np.matmul(T._hctransform, self._hctransform)

    def swap_axes_order(self) -> None:
        """Swap the axes order between x,y,z and z,y,x in place."""
        rev_ind = np.arange(self.NDIM, dtype=int)[::-1]
        A = np.identity(self.NDP1, dtype=float)
        A[
            # meshgrid returns a list, see numpy issue #23303
            tuple(np.meshgrid(rev_ind, rev_ind, indexing="ij", sparse=True, copy=False))
        ] = self._hctransform[: self.NDIM, : self.NDIM]
        A[rev_ind, self.NDIM] = self._hctransform[: self.NDIM, self.NDIM]
        self._hctransform = A

    def has_zero_translation(self) -> bool:
        """Check whether the translation vector is zero in this instance.

        Use numpy.allclose defaults for the testing tolerance.

        Returns
        -------
        True, if the translation vector equals close to zero.
        """
        return np.allclose(self._hctransform[: self.NDIM, self.NDIM], 0)

    def has_zero_offdiagonal(self) -> bool:
        """Check whether the off-diagonal elements of the transformation matrix are zero.

        Use numpy.allclose defaults for testing tolerance. Does not check the translation vector.
        """
        i_diag = np.arange(self.NDIM, dtype=int)
        A = np.array(self._hctransform[: self.NDIM, : self.NDIM], copy=True)
        A[i_diag, i_diag] = 0
        return np.allclose(A, 0)

    @staticmethod
    def create_from_ngff_entries(
        tr_entries: Iterable[Dict[str, npt.ArrayLike]],
        sl: Optional[slice] = None,
        ndim: int = 3,
    ) -> AffineTransformation:
        """Create transformation from an ome-ngff style flat iterable of dictionaries of array-like numbers.

        The transformations are applied in the order of presented by the iteration,
        ie. first transformation first.
        This is the OME-NGFF way of listing transformations.

        Does not alter the order of axes, ie. first axis will have index of 0.

        Parameters
        ----------
        sl: `slice` instance or similar that can be used as indexing
          Optional slice to apply to the numbers after conversion to ndarray.

        tr_entries: Iterable of dictionaries.
            Items must have a ``type`` key to describe type. Accordingly, array-like numbers must be under
             ``scale`` or ``translation`` keys.

        ndim: Number of dimensions
            Number of dimensions, passed to constructor.

        Returns
        -------
        New `AffineTransformation` instance.

        """
        T = AffineTransformation(ndim=ndim)
        for entry in tr_entries:
            if entry["type"] == "scale":
                T.left_compose(
                    AffineTransformation(scale=np.array(entry["scale"], copy=False)[sl], ndim=ndim)
                )
            elif entry["type"] == "translation":
                T.left_compose(
                    AffineTransformation(
                        translation=np.array(entry["translation"], copy=False)[sl],
                        ndim=ndim,
                    )
                )
            else:
                raise ValueError("Recognised transformation types are scale and translation")
        return T

    @staticmethod
    def create_from_ngff_multiscales(
        ngff_metadata: Dict[str, Any],
        i_multiscales: Optional[int] = 0,
        i_dataset: Optional[int] = 0,
        sl: Optional[slice] = None,
        ndim: int = 3,
    ) -> AffineTransformation:
        """Create transformation from an ome-ngff style multiscales json dictionary.

        Process the selected dataset transformations and the global transformations if present.

        Parameters
        ----------
        ngff_metadata:
          json dictionary
        i_multiscales:
          multiscales entry is a list. Index of this list to use, usually 0. Defaults to 0.
        i_dataset:
          Index of datasets in the selected
        """
        L = []
        # dataset transformations
        if "coordinateTransformations" in ngff_metadata["multiscales"][i_multiscales]["datasets"][i_dataset]:
            L.extend(
                ngff_metadata["multiscales"][i_multiscales]["datasets"][i_dataset][
                    "coordinateTransformations"
                ]
            )
        # global transformations
        if "coordinateTransformations" in ngff_metadata["multiscales"][i_multiscales]:
            L.extend(ngff_metadata["multiscales"][i_multiscales]["coordinateTransformations"])
        return AffineTransformation.create_from_ngff_entries(L, sl=sl, ndim=ndim)

    def get_as_ngff_transformations(
        self,
        ndim: Optional[int] = None,
        out_indices: Union[slice, npt.ArrayLike, None] = None,
    ) -> List[Dict[str, Any]]:
        """Return the current transformation as an ngff dictionary.

        Does not alter the order of axes, output in the order of ascending index.

        Raises
        ------
        ValueError: if the instance has non-zero off-diagonal transformation matrix elements

        Parameters
        ----------
        ndim:
          Number of dimensions for the output coordinateTransformations.
        out_indices:
          Object that can be used as indices for an ndim length array to set values.
          Typically slice or tuple of ints or array of ints.
        """
        L = []
        if ndim is None:
            ndim = self.NDIM

        if not self.has_zero_offdiagonal():
            raise ValueError("Non-zero off diagonals")

        entry = {"type": "scale"}
        s = np.ones(ndim, dtype=float)
        i_diag = np.arange(self.NDIM, dtype=int)
        s[out_indices] = self._hctransform[i_diag, i_diag]
        entry["scale"] = list(s)
        L.append(entry)

        entry = {"type": "translation"}
        s = np.zeros(ndim, dtype=float)
        s[out_indices] = self._hctransform[: self.NDIM, self.NDIM]
        entry["translation"] = list(s)
        L.append(entry)
        return L

    @staticmethod
    def create_from_xml_ViewRegistration(viewreg: ET.Element) -> AffineTransformation:
        """Parse a ViewRegistration xml element into an AffineTransformation instance.

        Parse all ViewTransform subelements and compose them into one transformation.
        Composition order is left-to-right, ie. the last transformation to be applied first.

        Does not alter the order of axes. Consider that BigDataViewer and ngff have reversed axis order.

        Returns
        -------
          New 3 dimensional instance. Defaults to identity if there are no ``ViewTransform`` entries.
        """
        T = AffineTransformation()
        for tr in viewreg.findall("ViewTransform"):
            if tr.get("type") != "affine":
                raise ValueError("Unknown transformation type, not affine.")
            aff = tr.find("affine")
            A = np.array([float(x) for x in aff.text.strip().split()], dtype=float).reshape(3, 4)
            T.right_compose(AffineTransformation(array=A[:, :3], translation=A[:, 3]))
        return T

    def get_as_xml_ViewTransform(self, name: Optional[str] = None) -> ET.Element:
        """Return the current transformation as a ViewTransform xml element.

        Does not alter the order of axes. Consider that BigDataViewer and ngff have reversed axis order.
        Instance must be 3 dimensional.

        Parameters
        ----------
        name:
          If provided, it will set the <Name> sub element. Otherwise <Name> will be missing.

        Returns
        -------
          New xml Element instance.
        """
        vt = ET.Element("ViewTransform")
        vt.attrib["type"] = "affine"
        if name is not None:
            x = ET.SubElement(vt, "Name")
            x.text = name

        x = ET.SubElement(vt, "affine")
        x.text = " ".join(map(str, self._hctransform[:3, :].reshape(12)))
        return vt

    @staticmethod
    def create_upscale_transformation(factor: int):
        """Create an coordinate transformation including the half pixel shifts.

        Create an affine transformation that scales the coordinates by the factor.
        """
        F = {1: 0, 2: 0.5, 4: 1.5, 8: 3.5, 16: 7.5, 32: 15.5, 64: 31.5}
        return AffineTransformation(
            scale=factor * np.ones(3, dtype=float), translation=F[factor] * np.ones(3, dtype=float)
        )

    @property
    def hctransform(self):
        """Property to access the internal homogeneous coordinates transformation matrix."""
        return self._hctransform

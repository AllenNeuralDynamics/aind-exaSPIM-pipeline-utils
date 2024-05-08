"""Unit test module for AffineTransformation class."""
import unittest
import numpy as np
import json
import xml.etree.ElementTree as ET
from aind_exaspim_pipeline_utils.qc.affine_transformation import AffineTransformation

JSON_TRANSFORMATION = """
{
    "multiscales": [
        {
            "axes": [
            ],
            "datasets": [
                {
                    "coordinateTransformations": [
                        {
                            "scale": [
                                1.0,
                                1.0,
                                1.0,
                                0.75,
                                0.75
                            ],
                            "type": "scale"
                        },
                        {
                            "translation": [
                                0,
                                0,
                                -28672.0,
                                -3979.3,
                                -5307.8
                            ],
                            "type": "translation"
                        }
                    ],
                    "path": "0"
                }
            ],
            "name": "/tile_x_0000_y_0000_z_0000_ch_561.zarr",
            "version": "0.4"
        }
    ]
}
"""

XML_TRANSFORMATION = """<?xml version="1.0" encoding="UTF-8"?>
<SpimData version="0.2">
  <BasePath type="relative">.</BasePath>
  <ViewRegistrations>
    <ViewRegistration timepoint="0" setup="0">
      <ViewTransform type="affine">
        <Name>AffineModel3D regularized with an RigidModel3D, lambda = 0.1</Name>
        <affine>1.0 0.0 0.0 0.0 0.0 2.0 0.0 -10.0 0.0 0.0 1.0 20.0</affine>
      </ViewTransform>
      <ViewTransform type="affine">
        <Name>TranslationModel3D</Name>
        <affine>1.0 0.0 0.0 -30.0 0.0 1.0 0.0 -40.0 0.0 0.0 1.0 30.0</affine>
      </ViewTransform>
      <ViewTransform type="affine">
        <Name>Translation to Nominal Grid</Name>
        <affine>1.0 0.0 0.0 -10.0 0.0 1.0 0.0 -20.0 0.0 0.0 1.0 -30.0</affine>
      </ViewTransform>
    </ViewRegistration>
  </ViewRegistrations>
  <BoundingBoxes />
  <PointSpreadFunctions />
  <StitchingResults />
  <IntensityAdjustments />
</SpimData>
"""


class TestAffineTransformation(unittest.TestCase):
    """Test cases for affine transformation representation."""

    def test_scaling(self):
        """Scaling."""
        A = AffineTransformation()
        # Test on array
        V1 = np.array([4, 5, 6], dtype=float)
        V2 = A.apply_to(V1)
        for x, y in zip(V1, V2):
            self.assertAlmostEqual(x, y, delta=0.001)

        A = AffineTransformation(scale=2)
        V2 = A.apply_to(V1)
        for x, y in zip(V2, (8, 10, 12)):
            self.assertAlmostEqual(x, y, delta=0.001)

    def test_translation(self):
        """Translation."""
        A = AffineTransformation(translation=(5, 6, 7))
        V1 = np.array([4, 5, 6], dtype=float)
        V2 = A.apply_to(V1)
        self.assertTrue(np.allclose(V2, [9, 11, 13]))
        B = AffineTransformation(scale=2)
        B.add_translation((5, 6, 7))
        V2 = B.apply_to(V1)
        self.assertTrue(np.allclose(V2, [13, 16, 19]))

    def test_multivector(self):
        """Application to multiple vectors."""
        B = AffineTransformation(scale=2, translation=(5, 6, 7))
        Vm = np.array([[1, 2, 3], [4, 5, 6], [4, 5, 6], [1, 2, 3]], dtype=float)
        Vm2 = B.apply_to(Vm)
        self.assertTrue(
            np.allclose(Vm2, [[7, 10, 13], [13, 16, 19], [13, 16, 19], [7, 10, 13]])
        )

    def test_composition(self):
        """Composition of transformations."""
        A = AffineTransformation(scale=5)
        B = AffineTransformation(translation=10)
        C = AffineTransformation(scale=5, translation=10)
        A.left_compose(B)
        self.assertTrue(np.allclose(C.get_matrix(), A.get_matrix()))
        self.assertTrue(np.allclose(C.get_translation(), A.get_translation()))

    def test_inverse(self):
        """Inverse."""
        A = AffineTransformation(
            array=[[10, 20, 30], [40, 50, 70], [70, 80, 90]], translation=[10, 20, 30]
        )
        B = A.get_inverse()
        C = B.copy()
        C.left_compose(A)
        self.assertTrue(np.allclose(C.get_matrix(), np.identity(3)))
        self.assertTrue(np.allclose(C.get_translation(), np.zeros(3)))

        C = B.copy()
        C.right_compose(A)
        self.assertTrue(np.allclose(C.get_matrix(), np.identity(3)))
        self.assertTrue(np.allclose(C.get_translation(), np.zeros(3)))

    def test_from_metadata(self):
        """Create from NGFF entries."""
        s1 = {"type": "scale", "scale": [2, 3, 4]}
        s2 = {"type": "translation", "translation": [10, 20, 30]}
        A = AffineTransformation.create_from_ngff_entries((s1, s2))
        B = AffineTransformation(scale=[2, 3, 4], translation=[10, 20, 30])
        self.assertTrue(np.allclose(A.get_matrix(), B.get_matrix()))
        self.assertTrue(np.allclose(A.get_translation(), B.get_translation()))

    def test_axes_swapping(self):
        """Swap axes."""
        A = AffineTransformation(
            array=[[10, 20, 30], [40, 50, 70], [70, 80, 90]], translation=[10, 20, 30]
        )
        v = np.array([4, 5, 6], dtype=float)
        vswap = v[::-1]
        vt1 = A.apply_to(v)
        A.swap_axes_order()
        vt2 = A.apply_to(vswap)
        self.assertTrue(np.allclose(vt1, vt2[::-1]))

    def test_json_parsing(self):
        """Parse and save as JSON string."""
        J = json.loads(JSON_TRANSFORMATION)
        A = AffineTransformation.create_from_ngff_multiscales(J, ndim=5)
        self.assertTrue(
            np.allclose(A.get_translation(), [0, 0, -28672.0, -3979.3, -5307.8])
        )

        A = AffineTransformation.create_from_ngff_multiscales(
            J, sl=slice(2, None), ndim=3
        )
        self.assertTrue(np.allclose(A.get_translation(), [-28672.0, -3979.3, -5307.8]))

        v = A.apply_to([2, 2, 2])
        self.assertTrue(
            np.allclose(v, [2 - 28672.0, 2 * 0.75 - 3979.3, 2 * 0.75 - 5307.8])
        )

        L = A.get_as_ngff_transformations()
        self.assertTrue(np.allclose(L[0]["scale"], [1.0, 0.75, 0.75]))

    def test_xml_parsing(self):
        """Parse and save as bdv XML."""
        r = ET.fromstring(XML_TRANSFORMATION)
        e = r.find("ViewRegistrations/ViewRegistration")
        A = AffineTransformation.create_from_xml_ViewRegistration(e)
        B = AffineTransformation(scale=[1, 2, 1], translation=[-40, -130, 20])
        self.assertTrue(np.allclose(A.get_matrix(), B.get_matrix()))
        self.assertTrue(np.allclose(A.get_translation(), B.get_translation()))

        L = B.get_as_xml_ViewTransform(name="Test Transformation")
        aff = L.find("affine")
        self.assertEqual(
            aff.text, "1.0 0.0 0.0 -40.0 0.0 2.0 0.0 -130.0 0.0 0.0 1.0 20.0"
        )


if __name__ == "__main__":
    unittest.main()

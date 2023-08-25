"""Tests for n5tozarr_da_convert functions."""
import unittest
from aind_exaspim_pipeline_utils.n5tozarr.n5tozarr_da import get_uri


class TestN5toZarr(unittest.TestCase):
    """N5toZarr tests"""

    def test_uri_formatting(self):
        """Uri string formatter test case"""
        s = get_uri("test_bucket", "/zarr_root/", "dataset_name")
        self.assertEqual(s, "s3://test_bucket/zarr_root/dataset_name/")

        s = get_uri(None, "/zarr_root/", "dataset_name")
        self.assertEqual(s, "/zarr_root/dataset_name/")

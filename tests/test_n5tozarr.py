"""Tests for n5tozarr_da_convert functions."""
import unittest
from aind_exaspim_pipeline_utils.n5tozarr.n5tozarr_da import fmt_uri


class TestN5toZarr(unittest.TestCase):
    """N5toZarr tests"""

    def test_uri_formatting(self):
        """Uri string formatter test case"""
        s = fmt_uri("s3:/test_bucket/zarr_root/dataset_name")
        self.assertEqual(s, "s3://test_bucket/zarr_root/dataset_name/")

        s = fmt_uri("/zarr_root/dataset_name")
        self.assertEqual(s, "/zarr_root/dataset_name/")

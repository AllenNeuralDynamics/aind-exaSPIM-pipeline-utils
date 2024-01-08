"""Tests for trigger capsule functions"""
import unittest
from datetime import datetime

import aind_exaspim_pipeline_utils.trigger.capsule


class TestCapsule(unittest.TestCase):
    """Capsule tests"""

    def test_get_fname_timestamp(self):
        """Test get_fname_timestamp"""
        timestamp = aind_exaspim_pipeline_utils.trigger.capsule.get_fname_timestamp(
            datetime(2020, 1, 1, 1, 2, 3))
        self.assertEqual(timestamp, "2020-01-01_01-02-03")

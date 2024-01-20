"""Tests for trigger capsule functions"""
import unittest
from datetime import datetime

import aind_exaspim_pipeline_utils.trigger.capsule


class TestCapsule(unittest.TestCase):
    """Capsule tests"""

    def test_get_fname_timestamp(self):
        """Test get_fname_timestamp"""
        timestamp = aind_exaspim_pipeline_utils.trigger.capsule.get_fname_timestamp(
            datetime(2020, 1, 1, 1, 2, 3)
        )
        self.assertEqual(timestamp, "2020-01-01_01-02-03")

    def test_recursive_update_mapping(self):
        """Test recursive_update_mapping"""
        mapping = {"a": 1, "b": {"c": 2, "d": 3}, "c": [{"a": 2}, {"a": 1}]}
        update = {"a": 4, "b": {"c": 5}, "c": [{}, {"a": 3}]}
        aind_exaspim_pipeline_utils.trigger.capsule.recursive_update_mapping(mapping, update)
        self.assertEqual(mapping["a"], 4)
        self.assertEqual(mapping["b"]["c"], 5)
        self.assertEqual(mapping["b"]["d"], 3)
        self.assertEqual(mapping["c"][0]["a"], 2)
        self.assertEqual(mapping["c"][1]["a"], 3)

        update2 = {"c": [{"a": 4}]}
        # This should raise ValueError
        with self.assertRaises(ValueError):
            aind_exaspim_pipeline_utils.trigger.capsule.recursive_update_mapping(mapping, update2)

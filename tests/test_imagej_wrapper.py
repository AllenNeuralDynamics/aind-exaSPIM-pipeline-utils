"""Tests for ImageJ wrapper functions and macro creator class"""
import contextlib
import io
import logging
import unittest
from unittest import mock

import argschema

from aind_exaspim_pipeline_utils.imagej_macros import ImagejMacros
from aind_exaspim_pipeline_utils.imagej_wrapper import (
    ImageJWrapperSchema,
    get_auto_parameters,
    wrapper_cmd_run,
)


class TestWrapperFunctions(unittest.TestCase):
    """Tests for indivicual functions"""

    @mock.patch("os.cpu_count")
    @mock.patch("psutil.virtual_memory")
    def testAutoParameters(self, mock_psutil_virtual_memory, mock_os_count):
        """Test for mem and cpu detection"""
        mock_os_count.return_value = 4
        mock_total = mock.Mock(total=128 * 1024 * 1024 * 1024)
        mock_psutil_virtual_memory.return_value = mock_total

        args = {"session_id": "test_session_123"}
        d = get_auto_parameters(args)

        self.assertIn("process_xml", d)
        self.assertEqual(d["auto_ncpu"], 4)
        self.assertEqual(d["auto_memgb"], 128 - 12)

    @mock.patch("subprocess.Popen")
    @mock.patch("selectors.DefaultSelector")
    def testCmdWrapper(self, mock_DefaultSelector, mock_subprocess_popen):
        """Tests for the cmd wrapper"""
        mock_selector = mock.Mock()
        mock_selector.configure_mock(**{"register.return_value": None, "close.return_value": None})
        mock_DefaultSelector.return_value.__enter__.return_value = mock_selector
        mock_std = mock.Mock()
        mock_std.configure_mock(**{"close.return_value": None, "read.return_value": b"text"})
        mock_popen = mock.Mock(stdout=mock_std, stderr=mock_std)
        mock_popen.configure_mock(**{"poll.return_value": 1, "wait.return_value": 0})
        mock_subprocess_popen.return_value.__enter__.return_value = mock_popen

        s_out = io.StringIO()
        s_err = io.StringIO()
        with contextlib.redirect_stdout(s_out), contextlib.redirect_stderr(s_err):
            r = wrapper_cmd_run("test_cmd", logging.getLogger())
        self.assertTrue("text" in s_err.getvalue())
        self.assertTrue("text" in s_out.getvalue())
        self.assertEqual(r, 1)
        mock_std.read.assert_called()
        mock_popen.poll.assert_called()


class TestMacros(unittest.TestCase):
    """Test case for ImagejMacros"""

    def setUp(self):
        """Set up ArgSchemaParser.args"""
        example_params_default = {
            "session_id": "2023-02-22",
            "memgb": 55,
            "parallel": 8,
            "dataset_xml": "test_dataset.xml",
            "do_detection": True,
            "ip_detection_params": {
                "downsample": 8,
                "bead_choice": "sample_small",  # Usual choice for the beads
                "set_minimum_maximum": True,
                "maximum_number_of_detections": 100000,
                "ip_limitation_choice": "around_median",
            },
            "do_registrations": True,
            "ip_registrations_params": [
                {
                    "transformation_choice": "translation",
                    "compare_views_choice": "overlapping_views",
                    "interest_point_inclusion_choice": "overlapping_ips",
                    "fix_views_choice": "first_fixed",
                    "fixed_tile_ids": [
                        10,
                    ],
                    "map_back_views_choice": "no_mapback",
                    "map_back_reference_view": 5,
                    "do_regularize": True,
                    "regularize_with_choice": "rigid",
                }
            ],
        }
        parser = argschema.ArgSchemaParser(
            schema_type=ImageJWrapperSchema, input_data=example_params_default, args=[]
        )
        self.args = parser.args

    def testMacroIPDet(self):
        """Test IP Detection macro"""

        det_params = dict(self.args["ip_detection_params"])
        det_params["process_xml"] = self.args["dataset_xml"]
        det_params["parallel"] = self.args["parallel"]

        m = ImagejMacros.get_macro_ip_det(det_params)
        self.assertRegex(m, "downsample_z=8x")
        self.assertNotRegex(m, "sigma=")

        det_params["bead_choice"] = "manual"
        m = ImagejMacros.get_macro_ip_det(det_params)
        self.assertRegex(m, "sigma=")
        print(m)

    def testMacroIPReg(self):
        """Test IP Registration macro"""
        reg_params = dict(self.args["ip_registrations_params"][0])
        reg_params["process_xml"] = self.args["dataset_xml"]
        reg_params["parallel"] = self.args["parallel"]
        m = ImagejMacros.get_macro_ip_reg(reg_params)
        self.assertRegex(m, "select=test_dataset.xml")
        self.assertNotRegex(m, "viewsetupid_")

        reg_params["fix_views_choice"] = "select_fixed"
        m = ImagejMacros.get_macro_ip_reg(reg_params)
        self.assertRegex(m, "viewsetupid_10_timepoint_0")

        reg_params["map_back_views_choice"] = "selected_translation"
        m = ImagejMacros.get_macro_ip_reg(reg_params)
        self.assertRegex(m, "ViewSetupId:5")

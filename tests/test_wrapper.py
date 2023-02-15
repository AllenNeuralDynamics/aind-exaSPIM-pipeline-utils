"""Tests for ImageJ wrapper functions and macro creator class"""
import contextlib
import io
import logging
import unittest
from unittest import mock

from aind_exaspim_pipeline_utils.imagej_macros import ImagejMacros
from aind_exaspim_pipeline_utils.imagej_wrapper import get_auto_parameters, wrapper_cmd_run


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
        self.assertEqual(d["ncpu"], 4)
        self.assertEqual(d["memgb"], 128 - 12)

    @mock.patch("subprocess.Popen")
    @mock.patch("selectors.DefaultSelector")
    def testCmdWrapper(self, mock_DefaultSelector, mock_subprocess_popen):
        """Tests for the cmd wrapper"""
        mock_selector = mock.Mock()
        mock_selector.configure_mock(**{"register.return_value": None, "close.return_value": None})
        mock_DefaultSelector.return_value = mock_selector
        mock_std = mock.Mock()
        mock_std.configure_mock(**{"close.return_value": None, "read.return_value": b"text"})
        mock_popen = mock.Mock(stdout=mock_std, stderr=mock_std)
        mock_popen.configure_mock(**{"poll.return_value": 1, "wait.return_value": 0})
        mock_subprocess_popen.return_value = mock_popen

        s_out = io.StringIO()
        s_err = io.StringIO()
        with contextlib.redirect_stdout(s_out), contextlib.redirect_stderr(s_err):
            r = wrapper_cmd_run("test_cmd", logging.getLogger())
        self.assertTrue("text" in s_err.getvalue())
        self.assertTrue("text" in s_out.getvalue())
        self.assertEqual(r, 0)
        mock_std.read.assert_called()
        mock_std.close.assert_called()
        mock_selector.close.assert_called()


class TestMacros(unittest.TestCase):
    """Test case for ImagejMacros"""

    def testMacroIPDet(self):
        """Test IP Detection macro"""
        m = ImagejMacros.get_macro_ip_det({"parallel": 4, "process_xml": "test_dataset.xml", "downsample": 8})
        self.assertRegex(m, "downsample_z=8x")

    def testMacroIPReg(self):
        """Test IP Registration macro"""
        m = ImagejMacros.get_macro_ip_reg({"parallel": 4, "process_xml": "test_dataset.xml"})
        self.assertRegex(m, "select=test_dataset.xml")

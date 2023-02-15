"""ImageJ Macro creator module."""
from typing import Dict


class ImagejMacros:
    """Generate imageJ macros from template strings by substituting parameter values.

    This class does not check substitution values and expects that all values are valid.
    """

    # TBD: Add interest_point_specification options
    MACRO_IP_DET = """
run("Memory & Threads...", "maximum=54000 parallel={parallel:d}");
run("Detect Interest Points for Registration",
 "select={process_xml} process_angle=[All angles] process_channel=[All channels] " +
"process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] " +
"type_of_interest_point_detection=Difference-of-Gaussian label_interest_points=beads " +
"subpixel_localization=[3-dimensional quadratic fit] " +
"interest_point_specification=[Comparable to Sample & small (beads)] " +
"downsample_xy=[Match Z Resolution (less downsampling)] downsample_z={downsample}x compute_on=[CPU (Java)]");
    """

    # TBD add these parameters to the schema
    MACRO_IP_REG = """
run("Memory & Threads...", "maximum=54000 parallel={parallel:d}");
run("Register Dataset based on Interest Points",
 "select={process_xml} process_angle=[All angles] process_channel=[All channels] " +
"process_illumination=[All illuminations] process_tile=[All tiles] " +
"process_timepoint=[All Timepoints] " +
"registration_algorithm=[Precise descriptor-based (translation invariant)] " +
"registration_in_between_views=[Only compare overlapping views (according to current transformations)] " +
"interest_point_inclusion=[Only compare interest points that overlap between views " +
"(according to current transformations)] " +
"interest_points=beads fix_views=[Fix first view] " +
"map_back_views=[Do not map back (use this if views are fixed)] " +
"transformation=Translation regularize_model model_to_regularize_with=Rigid " +
"lamba=0.10 number_of_neighbors=3 redundancy=1 significance=3 " +
"allowed_error_for_ransac=5 ransac_iterations=Normal");
    """
    # Fiji macro allows new line at , separated arguments
    # strings can be added " " + " "
    # within strings, ' ' can be used for file names with spaces

    @staticmethod
    def get_macro_ip_det(P: Dict) -> str:
        """Get IP detection macro.

        Parameters
        ----------
        P : `dict`
          Dictionary for macro template formatting.
        """
        return ImagejMacros.MACRO_IP_DET.format(**P)

    @staticmethod
    def get_macro_ip_reg(P: Dict) -> str:
        """Get IP registration macro.

        Parameters
        ----------
        P : `dict`
          Dictionary for macro template formatting.
        """
        return ImagejMacros.MACRO_IP_REG.format(**P)

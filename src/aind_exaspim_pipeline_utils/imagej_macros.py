"""ImageJ Macro creator module."""
from typing import Any, Dict


class ImagejMacros:
    """Generate imageJ macros from template strings by substituting parameter values.

    This class does not check substitution values and expects that all values are valid.
    """

    # Fiji macro allows new line at , separated arguments
    # strings can be added " " + " "
    # within strings, [ ] can be used for spaces, for file names? TBC

    MACRO_IP_DET = """
run("Memory & Threads...", "parallel={parallel:d}");
run("Detect Interest Points for Registration",
"select={process_xml} process_angle=[All angles] process_channel=[All channels] " +
"process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] " +
"type_of_interest_point_detection=Difference-of-Gaussian label_interest_points=beads " +
"subpixel_localization=[3-dimensional quadratic fit] " +
"interest_point_specification=[{bead_choice}] " +
"downsample_xy=[Match Z Resolution (less downsampling)] downsample_z={downsample}x{manual_bead_choice} " +
"compute_on=[CPU (Java)]{manual_minimum_maximum}{limit_amount_of_detections}");
    """

    MAP_BEAD_CHOICE = {
        "very_weak_small": "Very weak & small (beads)",
        "weak_small": "Weak & small (beads)",
        "sample_small": "Comparable to Sample & small (beads)",
        "strong_small": "Strong & small (beads)",
        "manual": "Advanced ...",
    }

    MAP_IP_LIMITATION_CHOICE = {
        "brightest": "Brightest",
        "around_median": "Around median (of those above threshold)",
        "weakest": "Weakest (above threshold)",
    }

    MACRO_IP_REG = """
run("Memory & Threads...", "parallel={parallel:d}");
run("Register Dataset based on Interest Points",
"select={process_xml} process_angle=[All angles] process_channel=[All channels] " +
"process_illumination=[All illuminations] process_tile=[All tiles] " +
"process_timepoint=[All Timepoints] " +
"registration_algorithm=[Precise descriptor-based (translation invariant)] " +
"registration_in_between_views=[{compare_views}] " +
"interest_point_inclusion=[{interest_point_inclusion}] " +
"interest_points=beads fix_views=[{fix_views}] " +
"map_back_views=[{map_back_views}] " +
"transformation={transformation}{regularization} " +
"number_of_neighbors=3 redundancy=1 significance=3 " +
"allowed_error_for_ransac=5 ransac_iterations=Normal{fixed_viewsetupids}{select_reference_views}");
"""

    TEMPLATE_REGULARIZE = """ regularize_model model_to_regularize_with={regularize_with} lambda=0.10"""

    MAP_COMPARE_VIEWS = {
        "all_views": "Compare all views against each other",
        "overlapping_views": "Only compare overlapping views (according to current transformations)",
    }
    MAP_INTEREST_POINT_INCLUSION = {
        "overlapping_ips": "Only compare interest points that overlap "
        + "between views (according to current transformations)",
        "all_ips": "Compare all interest point of overlapping views",
    }

    MAP_FIX_VIEWS = {
        "no_fixed": "Do not fix views",
        "first_fixed": "Fix first view",
        "select_fixed": "Select fixed view",
    }

    MAP_MAP_BACK_VIEWS = {
        "no_mapback": "Do not map back (use this if views are fixed)",
        "first_translation": "Map back to first view using translation model",
        "first_rigid": "Map back to first view using rigid model",
        "selected_translation": "Map back to user defined view using translation model",
        "selected_rigid": "Map back to user defined view using rigid model",
    }

    MAP_TRANSFORMATION = {"translation": "Translation", "rigid": "Rigid", "affine": "Affine"}

    MAP_REGULARIZATION = {
        "identity": "Identity",
        "translation": "Translation",
        "rigid": "Rigid",
        "affine": "Affine",
    }

    @staticmethod
    def get_macro_ip_det(P: Dict[str, Any]) -> str:
        """Get a parameter formatted IP detection macro.

        Parameters
        ----------
        P : `dict`
          Parameter dictionary for macro formatting.
        """
        fparams = dict(P)
        fparams["manual_bead_choice"] = ""
        if fparams["bead_choice"] == "manual":
            fparams["find_minima"] = " find_minima" if P["find_minima"] else ""
            fparams["find_maxima"] = " find_maxima" if P["find_maxima"] else ""
            fparams[
                "manual_bead_choice"
            ] = " sigma={sigma:.5f} threshold={threshold:.5f}{find_minima}{find_maxima}".format(**fparams)
        fparams["bead_choice"] = ImagejMacros.MAP_BEAD_CHOICE[P["bead_choice"]]

        fparams["manual_minimum_maximum"] = ""
        if fparams["set_minimum_maximum"]:
            fparams["manual_minimum_maximum"] = (
                " set_minimal_and_maximal_intensity minimal_intensity={minimal_intensity:.1f}"
                " maximal_intensity={maximal_intensity:.1f}"
            ).format(**fparams)

        fparams["limit_amount_of_detections"] = ""
        if P["maximum_number_of_detections"] > 0:
            fparams["limit_amount_of_detections"] = " limit_amount_of_detections maximum_number={:d}".format(
                P["maximum_number_of_detections"]
            ) + " type_of_detections_to_use=[{}]".format(
                ImagejMacros.MAP_IP_LIMITATION_CHOICE[P["ip_limitation_choice"]]
            )
        return ImagejMacros.MACRO_IP_DET.format(**fparams)

    @staticmethod
    def get_macro_ip_reg(P: Dict[str, Any]) -> str:
        """Get a parameter formatted IP registration macro.

        Parameters
        ----------
        P : `dict`
          Parameter dictionary for macro template formatting.
        """
        fparams = dict(P)
        fparams["compare_views"] = ImagejMacros.MAP_COMPARE_VIEWS[P["compare_views_choice"]]
        fparams["interest_point_inclusion"] = ImagejMacros.MAP_INTEREST_POINT_INCLUSION[
            P["interest_point_inclusion_choice"]
        ]
        fparams["fix_views"] = ImagejMacros.MAP_FIX_VIEWS[P["fix_views_choice"]]
        fparams["transformation"] = ImagejMacros.MAP_TRANSFORMATION[P["transformation_choice"]]
        fparams["map_back_views"] = ImagejMacros.MAP_MAP_BACK_VIEWS[P["map_back_views_choice"]]
        fparams["regularization"] = ""
        fparams["fixed_viewsetupids"] = ""
        if P["do_regularize"]:
            fparams["regularization"] = ImagejMacros.TEMPLATE_REGULARIZE.format(
                regularize_with=ImagejMacros.MAP_REGULARIZATION[P["regularize_with_choice"]]
            )
        if P["fix_views_choice"] == "select_fixed":
            fparams["fixed_viewsetupids"] = "".join(
                f" viewsetupid_{tile_id:d}_timepoint_0" for tile_id in P["fixed_tile_ids"]
            )
        fparams["select_reference_views"] = ""
        if P["map_back_views_choice"] in ("selected_translation", "selected_rigid"):
            fparams[
                "select_reference_views"
            ] = " select_reference_views=[ViewSetupId:{:d} Timepoint:0]".format(P["map_back_reference_view"])
        return ImagejMacros.MACRO_IP_REG.format(**fparams)

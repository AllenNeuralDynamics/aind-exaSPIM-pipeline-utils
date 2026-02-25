#!/usr/bin/env python
"""
Small debug harness for aind_exaspim_pipeline_utils.imagej_wrapper.
"""

import json
import logging
import shutil
import os

import argschema

from aind_exaspim_pipeline_utils.imagej_wrapper import (
    ImageJWrapperSchema,
    get_auto_parameters,
    wrapper_cmd_run,
)
from aind_exaspim_pipeline_utils.imagej_macros import ImagejMacros

# ---- Point directly to your Fiji / ImageJ binary ---------------------------

IMAGEJ_BIN = "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"

# ---- Hard-coded config -----------------------------------------------------

EXAMPLE_CONFIG = {
    "session_id": "HCR_823476-s1-ls2_2025-11-18_00-00-00",
    "memgb": 100,
    "parallel": 16,
    "dataset_xml": "/Users/sean.fite/Desktop/bigstitcher.xml",
    "do_phase_correlation": True,
    "do_detection": False,
    "do_registrations": False,
    "phase_correlation_params": {
        "downsample": 4,
        "min_correlation": 0.6,
        "max_shift_in_x": 0,
        "max_shift_in_y": 0,
        "max_shift_in_z": 0,
    },
}

RESULTS_DIR = os.path.expanduser("~/Desktop/results")


def main(run_imagej: bool = False) -> None:
    """Run a single phase-correlation debug pass."""
    logger = logging.getLogger(__name__)

    # Parse config via argschema, like the real wrapper
    parser = argschema.ArgSchemaParser(
        schema_type=ImageJWrapperSchema,
        input_data=EXAMPLE_CONFIG,
        args=[],
    )
    args = dict(parser.args)
    print("Parsed args from EXAMPLE_CONFIG:")
    print(json.dumps(args, indent=2, default=str))

    # Auto params (process_xml, macro paths, mem, etc.)
    args.update(get_auto_parameters(args))

    # Override ../results/ to your Desktop/results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    args["process_xml"] = os.path.join(RESULTS_DIR, "bigstitcher.xml")
    args["macro_phase_corr"] = os.path.join(RESULTS_DIR, "macro_phase_corr.ijm")
    args["macro_ip_det"] = os.path.join(RESULTS_DIR, "macro_ip_det.ijm")

    print("Args after get_auto_parameters + overrides:")
    print(json.dumps(args, indent=2, default=str))

    # Write config next to everything else
    cfg_path = os.path.join(RESULTS_DIR, "config_debug.json")
    print(f"Writing config_debug.json -> {cfg_path}")
    with open(cfg_path, "w") as f:
        json.dump(args, f, indent=2, default=str)

    # Copy XML into the working location
    print(f"Copying input xml {args['dataset_xml']} -> {args['process_xml']}")
    shutil.copy(args["dataset_xml"], args["process_xml"])

    # Phase correlation branch
    if args.get("do_phase_correlation"):
        phase_params = dict(args["phase_correlation_params"])
        phase_params["process_xml"] = args["process_xml"]
        phase_params["parallel"] = args["parallel"]

        # Extra params needed by MACRO_PROTEOMICS_PHASE_CORRELATION
        phase_params["relative_optimization_threshold"] = 2.5
        phase_params["absolute_optimization_threshold"] = 3.5

        # 🔄 Use the iterative-dropping strategy macro
        macro_body = ImagejMacros.get_macro_proteomics_phase_correlation(phase_params)

        print("=== Proteomics phase correlation (iterative) macro BEGIN ===")
        print(macro_body)
        print("=== Proteomics phase correlation (iterative) macro END ===")

        print(f"Creating macro file {args['macro_phase_corr']}")
        with open(args["macro_phase_corr"], "w") as f:
            f.write(macro_body)

        if run_imagej:
            print(f"Running ImageJ for phase correlation using: {IMAGEJ_BIN}")
            r = wrapper_cmd_run(
                [
                    IMAGEJ_BIN,
                    "-Dimagej.updater.disableAutocheck=true",
                    "--headless",
                    "--memory",
                    "{memgb}G".format(**args),
                    "--console",
                    "--run",
                    args["macro_phase_corr"],
                ],
                logger,
            )
            print(f"ImageJ return code: {r}")
            if r != 0:
                raise RuntimeError(f"Phase Correlation command failed with code {r}")
        else:
            print("run_imagej=False; not launching ImageJ (macro generation only).")

    print("Debug script finished.")


if __name__ == "__main__":
    # Switch this to False if you only want macro generation
    main(run_imagej=True)
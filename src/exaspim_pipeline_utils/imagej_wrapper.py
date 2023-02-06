import os
import shutil
import subprocess
import sys
import logging
import selectors
import psutil
import argschema
from typing import List, Union, Dict

import psutil
from argschema.fields import Boolean, Nested, String, Int
from .imagej_macros import ImagejMacros


class IPDetectionSchema(argschema.ArgSchema):
    """Adjustable parameters to detect IP."""
    downsample = Int(required=True,
                     metadata={'description': 'Downsampling factor. Use the one that is available in the dataset.'})
    # TBD: sigma1, sigma2

class IPRegistrationSchema(argschema.ArgSchema):
    """Adjustable parameters to register with translation only."""
    downsample = Int(required=True,
                     metadata={'description': 'Downsampling factor. Use the one that is available in the dataset.'})
    # TBD:  Fixed tile numbers


class ImageJWrapperSchema(argschema.ArgSchema):
    """Command line arguments.
    """
    session_id = String(required=True, metadata={'description': 'Processing run session identifier'})
    dataset_xml = String(required=True, metadata={'description': 'Input xml dataset definition'})
    do_detection = Boolean(required=True, metadata={'description': 'Do interest point detection?'})
    ip_detection_params = Nested(IPDetectionSchema, required=False,
                                 metadata={'description': 'Interest point detection parameters'
                                           })
    do_registration = Boolean(required=True,
                              metadata={'description': 'Do transformation fitting (translation only at the moment) ?'
                                        })
    ip_registration_params = Nested(IPRegistrationSchema, required=False,
                                    metadata={'description': 'Registration parameters'
                                              })


def wrapper_cmd_run(cmd: Union[str, List], logger:logging.Logger) -> Int:
    """
    Wrapper for a shell command.

    Wraps a shell command.

    It captures and re-prints stdout and strderr as the command progresses.
    TBD: run validation while the program is running and kill it if failure detected.

    Parameters
    ----------
    cmd: `str`
        Command that we want to execute.

    Returns
    -------
    r: `int`
      Cmd return code.
    """
    logger.info("Starting command (%s)", str(cmd))
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )

    sel = selectors.DefaultSelector()
    try:
        sel.register(p.stdout, selectors.EVENT_READ)
        sel.register(p.stderr, selectors.EVENT_READ)

        while True:
            for key, _ in sel.select():
                data = key.fileobj.read1().decode()
                if not data:
                    break
                if key.fileobj is p.stdout:
                    print(data, end="")
                else:
                    print(data, end="", file=sys.stderr)
    finally:
        p.stdout.close()
        p.stdin.close()
        sel.close()
    r = p.wait()
    logger.info("Command finished with return code %d", r)
    return r


def get_auto_parameters(args: Dict) -> Dict:
    """Determine environment parameters.

    Determine number of cpus, imagej memory limit and imagej macro file names.

    Parameters
    ----------
    args: `Dict`
        ArgSchema args dictionary

    Returns
    -------
    params: `Dict`
      New dictionary with determined parameters.

    """
    ncpu = os.cpu_count()

    mem_GB = psutil.virtual_memory().total // (1024 * 1024 * 1024)
    d = int(mem_GB * 0.1)
    if d < 5:
        d = 5
    mem_GB -= d
    if mem_GB < 10:
        raise ValueError("Too little memory available")

    process_xml = "/results/bigstitcher_{session_id}.xml".format(args)
    macro_ip_det = "/results/macro_ip_det_{session_id}.ijm".format(args)
    macro_ip_reg = "/results/macro_ip_reg_{session_id}.ijm".format(args)
    return {'process_xml': process_xml, 'ncpu': ncpu, 'memgb': mem_GB, 'macro_ip_det': macro_ip_det,
            'macro_ip_reg': macro_ip_reg}



def main():
    """Entry point if run as a standalone program.
    """
    logging.basicConfig(format='%(asctime)s %(levelname)-7s %(name)s %(message)s')
    logger = logging.getLogger()
    parser = argschema.ArgSchemaParser(schema_type=ImageJWrapperSchema)
    args = dict(parser.args)
    args.update(get_auto_parameters(args))

    logger.info("Copying input xml %s -> %s", args['dataset_xml'], args['process_xml'])
    shutil.copy(args['dataset_xml'], args['process_xml'])

    if args['do_detection']:
        det_params = dict(args['ip_detection_params'])
        det_params['parallel'] = args['ncpu']
        det_params['process_xml'] = args['process_xml']
        logger.info("Creating macro %s", args['macro_ip_det'])
        with open(args['macro_ip_det'], 'w') as f:
            f.write(ImagejMacros.get_macro_ip_det(det_params))
        wrapper_cmd_run(["ImageJ", "-Dimagej.updater.disableAutocheck=true", "--headless", "--memory", "{memgb}G".format(args),
                         "--console", "--run", args['macro_ip_det']], logger)

    if args['do_registration']:
        reg_params = dict(args['ip_registration_params'])
        reg_params['parallel'] = args['ncpu']
        reg_params['procsess_xml'] = args['process_xml']
        logger.info("Creating macro %s", args['macro_ip_det'])
        with open(args['macro_ip_reg'], 'w') as f:
            f.write(ImagejMacros.get_macro_ip_reg(reg_params))
        wrapper_cmd_run(
            ["ImageJ", "-Dimagej.updater.disableAutocheck=true", "--headless", "--memory", "{memgb}G".format(args),
             "--console", "--run", args['macro_ip_reg']], logger)
    logger.info("Done.")

if __name__ == "__main__":
    main()
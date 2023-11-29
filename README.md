# exaSPIM pipeline utils

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)

Code repository to be installed in exaSPIM processing capsules.

## Features

 - Wrapper code for ImageJ automation.
 - n5 to zarr converter to be run in a Code Ocean capsule.

### ImageJ wrapper module

The ImageJ wrapper module contains Fiji macro templates and wrapper code to 
automatically run interest point detection and interest point based registration
in the Code Ocean capsule. This functionality is set as the main entry point of
the package if the whole package is invoked on the command line or the 
`aind_exaspim_pipeline` command is run.

```bash
#!/usr/bin/env bash
set -ex
cd ~/capsule
imagej_wrapper "$@"
```

### N5 to Zarr converter

The N5 to zarr converter sets up a local dask cluster with multiple python processes 
as workers to read in an N5 dataset and write it out in a multiscale Zarr dataset.
Both datasets may be local or directly on S3. AWS credentials must be available in the
environment (Code Ocean credential assignment to environment variables).

This implementation is based on dask.array (da).

This command takes a manifest json file as the only command line argument or looks it 
up at the hard-wired `data/manifest/exaspim_manifest.json` location if not specified.

To set up a code ocean capsule, use the following `run.sh` script:

```bash
#!/usr/bin/env bash
set -ex
cd ~/capsule
n5tozarr_da_converter "$@"
```


## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e .[dev]
```

For n5tozarr and zarr multiscale conversion, install as
```bash
pip install -e .[n5tozarr]
```

## Contributing

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```bash
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```bash
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```bash
flake8 .
```

- Use **black** to automatically format the code into PEP standards:
```bash
black .
```

- Use **isort** to automatically sort import statements:
```bash
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Documentation
To generate the rst files source files for documentation, run
```bash
sphinx-apidoc -o doc_template/source/ src 
```
Then to create the documentation HTML files, run
```bash
sphinx-build -b html doc_template/source/ doc_template/build/html
```
More info on sphinx installation can be found [here](https://www.sphinx-doc.org/en/master/usage/installation.html).

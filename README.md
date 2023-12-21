## preprocessing
`preprocessing` is a python library designed for the purpose of preprocessing MRI data at QTIM. It currently supports reorganization of dicom files, subsequent nifti conversion, and preprocessing for brain data. Its outputs are intended to follow the BIDS organizational scheme.

## Table of Contents
* **[Installation Guide](#installation-guide)**
   * [Python Installation](#python-installation)
   * [External Software](#external-software)
* **[CLI User Guide](#cli-user-guide)**
   * [old_project_anon_keys Command](old_project_anon_keys-command)

## Installation Guide
If you are working on a Martinos machine and want to use `preprocessing` directly, you will already have access to it through the corresponding pyenv environment (visit this [guide](https://github.com/QTIM-Lab/qtim-standards/blob/main/environment_setup.md) for more information on using pyenv at Martinos). Simply run: 
```bash
pyenv activate preprocessing
```
and you will be able to use all of the functionality provided through the CLI.

If however, you are working on another machine or want to use the `preprocessing` library as part of a larger project, you should install it into a python virtual environment alongside some external software. 

### Python Installation
To install the `preprocessing` library in a python virtual environment, you may use either pip or poetry, like so:
```bash
pip install git+ssh://git@github.com/QTIM-Lab/preprocessing.git
```

```bash
poetry add git+ssh://git@github.com/QTIM-Lab/preprocessing.git
```

### External Software
Aside from python dependencies, this repo also requires external software to run some of the commands. On Martinos Machines, everything will be sourced automatically and will require no additional work on your part. If you want to use `preprocessing` on your own machine, you will have to install ANTS, Slicer, and dcm2niix (easily available through fsl) and source them within your shell. For user-specific or otherwise non-system installations, it is recommended to add analogous lines to the following directly to your .bashrc, .zshrc, etc.:
```bash
export PATH=/usr/pubsw/packages/fsl/6.0.6/bin:${PATH}
export PATH=/usr/pubsw/packages/slicer/Slicer-5.2.2-linux-amd64/:${PATH}
export PATH=/usr/pubsw/packages/ANTS/2.3.5/bin:${PATH}

export ANTSPATH=/usr/pubsw/packages/ANTS/2.3.5/bin
export FSLDIR=/usr/pubsw/packages/fsl/6.0.6/
source ${FSLDIR}/etc/fslconf/fsl.sh
```

### CLI User Guide
While you can import this code directly into your own python scripts, the most convenient way to run common functions is through the CLI. Once installed to your project's environment using pip or poetry, you will be able to access the most important functionality through `preprocessing`'s various subcommands. At any time, use `preprocessing --help` for an overview of the available commands and a brief description of each. For more in depth descriptions of each command, use `preprocessing <command> --help`.

This library utilizes csv files for the majority of its functions. Specific columns may be required for each to run properly. If you are using the CLI, use `--help` to view the columns required for a given command. Alternatively, reference the docstrings if you wish to use `preprocessing` directly within python. For an example of what formatting to expect, check [here](example.csv).

## old_project_anon_keys Command
```
preprocessing old_project_anon_keys <input_dir> <output_dir>
```
This command creates anonymization keys for anonymous PatientID and StudyID from the previous QTIM organizational scheme. Is compatible with data following a following <Patient_ID>/<Study_ID> directory hierarchy.

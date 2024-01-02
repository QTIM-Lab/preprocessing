## preprocessing
`preprocessing` is a python library designed for the purpose of preprocessing MRI data at QTIM. It currently supports reorganization of dicom files, subsequent nifti conversion, and preprocessing for brain data. Its outputs are intended to follow the BIDS organizational scheme.

## Table of Contents
* **[Installation Guide](#installation-guide)**
   * [Python Installation](#python-installation)
   * [External Software](#external-software)
* **[CLI User Guide](#cli-user-guide)**
   * [old-project-anon-keys Command](#old-project-anon-keys-command)
   * [reorganize-dicoms Command](#reorganize-dicoms-command)
   * [dataset-to-nifti Command](#dataset-to-nifti-command)
   * [predict-series Command](#predict-series-command)
   * [brain-preprocessing Command](#brain-preprocessing-command)

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

## CLI User Guide
While you can import this code directly into your own python scripts, the most convenient way to run common functions is through the CLI. Once installed to your project's environment using pip or poetry, you will be able to access the most important functionality through `preprocessing`'s various subcommands. At any time, use `preprocessing --help` for an overview of the available commands and a brief description of each. For more in depth descriptions of each command, use `preprocessing <command> --help`.

This library utilizes csv files for the majority of its functions. Specific columns may be required for each to run properly. If you are using the CLI, use `--help` to view the columns required for a given command. Alternatively, reference the docstrings if you wish to use `preprocessing` directly within python. For an example of what formatting to expect, check [here](example.csv).

### old-project-anon-keys Command
```bash
preprocessing old-project-anon-keys <input-dir> <output-dir>
```
This command creates anonymization keys for anonymous PatientID and StudyID from the previous QTIM organizational scheme. This command is compatible with data following a following <Patient_ID>/<Study_ID> directory hierarchy.

### reorganize-dicoms Command
```bash
preprocessing reorganize-dicoms <original-dicom-dir> <new-dicom-dir> \
        [--anon-csv=None] [-c | --cpus=0]
```
This command reorganizes DICOMs to follow the BIDS convention. Any DICOMs found recursively within this directory will be reorganized (at least one level of subdirectories is assumed). Anonomyzation keys for PatientIDs and StudyIDs are provided within a csv.

### dataset-to-nifti Command
```bash
preprocessing dataset-to-nifti <nifti-dir> <csv> [--overwrite=False] \
        [-c | --cpus=0] [-h | --help]
```
This command converts DICOMs to NIfTI file format. A csv is required to map a DICOM series to the resulting .nii.gz file and to provide the context for filenames. The outputs will comply with the BIDS conventions.

### predict-series Command
```bash
preprocessing predict-series <csv> [--ruleset="brain"] \
        [--description-key=default_key] [-c | --cpus=0] \
        [-h | --help]
```
This command predicts the sequence type for every series in your dataset. A csv is required to indicate the location of the corresponding DICOMs. Predictions are made using the mr-series-selection repo's analysis of the DICOM header. A json can be provided to combine multiple NormalizedDescriptions into a single category.
Description keys should map a NormalizedDescription to a standard name and a SeriesType (BIDS modality) like so:

default_key = {\
&nbsp;&nbsp;&nbsp;&nbsp;"T1Pre": [["iso3D AX T1 NonContrast", "iso3D AX T1 NonContrast RFMT"], "anat"],\
&nbsp;&nbsp;&nbsp;&nbsp;"T1Post": [["iso3D AX T1 WithContrast", "iso3D AX T1 WithContrast RFMT"], "anat"],\
}

### brain-preprocessing Command
```bash
preprocessing brain-preprocessing <preprocessed-dir> <csv> \
        [--pipeline-key="preprocessed"] [--registration-key="T1Post"] \
        [--longitudinal-registration=False] [--orientation="RAI"] \
        [--spacing="1,1,1"] [--no-skullstrip=False] [-c | --cpus=0] \
        [-v | --verbose=False] [-h | --help]
```
This command preprocesses NIfTI files for deep learning. A csv is required to indicate the location of source files and to procide the context for filenames. The outputs will comply with BIDS conventions.

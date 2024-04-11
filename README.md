## preprocessing
`preprocessing` is a python library designed for the purpose of preprocessing MRI data at QTIM. It currently supports reorganization of dicom and nifti files to follow BIDS conventions, dicom to nifti conversion, and preprocessing for brain data. Its outputs are intended to follow the BIDS organizational scheme.

## Table of Contents
* **[Installation Guide](#installation-guide)**
   * [Python Installation](#python-installation)
   * [External Software](#external-software)
   * [External Models](#external-models)
* **[CLI User Guide](#cli-user-guide)**
   * [old-project-anon-keys Command](#old-project-anon-keys-command)
   * [nifti-dataset-anon-keys Command](#nifti-dataset-anon-keys-command)
   * [reorganize-dicoms Command](#reorganize-dicoms-command)
   * [dataset-to-nifti Command](#dataset-to-nifti-command)
   * [predict-series Command](#predict-series-command)
   * [brain-preprocessing Command](#brain-preprocessing-command)
   * [track-tumors Command](#track-tumors-command)

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
Aside from python dependencies, this repo also requires external software to run some of the commands. On Martinos Machines, everything will be sourced automatically and will require no additional work on your part. If you want to use `preprocessing` on your own machine, you will have to install dcm2niix and source it within your shell. It is easily accessible as a part of `fsl`. For user-specific or otherwise non-system installations, it is recommended to add analogous lines to the following directly to your .bashrc, .zshrc, etc.:
```bash
export PATH=/usr/pubsw/packages/fsl/6.0.6/bin:${PATH}
export FSLDIR=/usr/pubsw/packages/fsl/6.0.6/
source ${FSLDIR}/etc/fslconf/fsl.sh
```

### External Models
In order to complete skullstripping and registration tasks, `preprocessing` relies on [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) and [SynthMorph](https://martinos.org/malte/synthmorph/). The first time a command that requires these models is called, you will be prompted to define an environment variable `PREPROCESSINF_MODELS_PATH` and to update your RC file to be used in the future. If you are on a Martinos machine, these models are already downloaded and available if you specify 'Martinos'.

## CLI User Guide
While you can import this code directly into your own python scripts, the most convenient way to run common functions is through the CLI. Once installed to your project's environment using pip or poetry, you will be able to access the most important functionality through `preprocessing`'s various subcommands. At any time, use `preprocessing --help` for an overview of the available commands and a brief description of each. For more in depth descriptions of each command, use `preprocessing <command> --help`.

This library utilizes csv files for the majority of its functions. Specific columns may be required for each to run properly. If you are using the CLI, use `--help` to view the columns required for a given command. Alternatively, reference the docstrings if you wish to use `preprocessing` directly within python. For an example of what formatting to expect, check [here](example.csv).

### old-project-anon-keys Command
```bash
preprocessing old-project-anon-keys <input-dir> <output-dir> \
        [-h | --help]
```
This command creates anonymization keys for anonymous PatientID and StudyID from the previous QTIM organizational scheme. This command is compatible with data following a following <Patient_ID>/<Study_ID> directory hierarchy.

### nifti-dataset-anon-keys Command
```bash
preprocessing nifti-dataset-anon-keys <nifti-dir> <output-dir> \
        [--normalized-descriptions] [-h | --help]
```
This command creates anonymization keys for a dataset that starts within NIfTI format. If the 'SeriesDescription's are not normalized, 'NormalizedSeriesDescription's must be obtained externally before the NIfTI dataset can be reorganized.

### reorganize-dicoms Command
```bash
preprocessing reorganize-dicoms <original-dicom-dir> <new-dicom-dir> \
        [--anon-csv] [-c | --cpus=1] [-h | --help]
```
This command reorganizes DICOMs to follow the BIDS convention. Any DICOMs found recursively within this directory will be reorganized (at least one level of subdirectories is assumed). Anonomyzation keys for PatientIDs and StudyIDs are provided within a csv.

### dataset-to-nifti Command
```bash
preprocessing dataset-to-nifti <nifti-dir> <csv> [--overwrite] \
        [-c | --cpus=1] [-h | --help]
```
This command converts DICOMs to NIfTI file format. A csv is required to map a DICOM series to the resulting .nii.gz file and to provide the context for filenames. The outputs will comply with the BIDS conventions.

### predict-series Command
```bash
preprocessing predict-series <csv> [--ruleset="brain"] \
        [--description-key=default_key] [-c | --cpus=1] \
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
        [-p | --patients] [-pk | --pipeline-key="preprocessed"] \
        [-rk | --registration-key="T1Post"] \
        [-l | --longitudinal-registration=False] [-a | --atlas-target] \
        [-m | --model="affine"] [ -o | --orientation="RAS"] \
        [-s | --spacing="1,1,1"] [-ns | --no-skullstrip=False] \
        [-ps | --pre-skullstripped] [-b | --binarize-seg] \
        [-c | --cpus=1] [-g | --gpu=False] [-v | --verbose=False] \
        [-d | --debug] [-h | --help]
```
This command preprocesses NIfTI files for deep learning. A csv is required to indicate the location of source files and to procide the context for filenames. The outputs will comply with BIDS conventions.

### track-tumors Command
```
preprocessing track-tumors <tracking-dir> <csv> \
        [-p | patients] [-pk | --pipeline-key="preprocessed"] \
        [-l | --labels="1"] [-c | --cpus=1] [-h | --help]
```

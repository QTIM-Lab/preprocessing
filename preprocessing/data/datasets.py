"""
The `datasets` module contains tools for constructing DICOM and NIfTI
datasets from a collection of files in the corresponding format.

Public Functions
----------------
anonymize_df
    Apply automated anonymization to a DataFrame. This function assumes
    that the 'PatientID' and 'StudyID' tags are consistent and correct
    to derive AnonPatientID = 'sub_{i:02d}' and AnonStudyID = 'ses_{i:02d}'.

create_dicom_dataset
    Create a DICOM dataset CSV compatible with subsequent `preprocessing`
    scripts. The final CSV provides a series level summary of the location
    of each series alongside metadata extracted from DICOM headers.  If the
    previous organization schems of the dataset does not enforce a DICOM
    series being isolated to a unique directory (instances belonging to
    multiple series must not share the same lowest level directory),
    reorganization must be applied for NIfTI conversion.

create_nifti_dataset
    Create a NIfTI dataset CSV compatible with subsequent `preprocessing`
    scripts. The final CSV provides a series level summary of the location
    of each series alongside metadata generated to simulate DICOM headers.
    Specifically, ['PatientID', 'StudyDate', 'SeriesInstanceUID',
    'SeriesDescription', 'StudyInstanceUID'] (and optionally
    'NormalizedSeriesDescription') are inferred or randomly generated.
"""
import pandas as pd
import datetime
import csv

from pathlib import Path
from tqdm import tqdm
from concurrent.futures import as_completed, ProcessPoolExecutor
from typing import Literal, Callable, Sequence, Tuple, Dict, Any
from preprocessing.utils import (
    parse_string,
    update_errorfile,
    check_required_columns,
    cglob
)
from preprocessing.constants import META_KEYS, REQUIRED_KEYS
from pydicom.uid import generate_uid
from pydicom import dcmread, config


def anonymize_df(df: pd.DataFrame, check_columns: bool = True):
    """
    Apply automated anonymization to a DataFrame. This function assumes
    that the 'PatientID' and 'StudyID' tags are consistent and correct
    to derive 'AnonPatientID' = 'sub_{i:02d}' and 'AnonStudyID' = 'ses_{i:02d}'.

    Parameters
    ----------
    df: pd.DataFrame
        A DataFrame for which you wish to provide anonymized patient and study
        identifiers. It must contain the columns: 'PatientID' and 'StudyDate'.

    check_columns: bool
        Whether to check `df` for required columns. Defaults to True.

    Returns
    -------
    pd.DataFrame
        `df` with added or corrected columns 'AnonPatientID' and
        'AnonStudyID'.
    """

    if check_columns:
        required_columns = [
            "PatientID",
            "StudyDate",
            "SeriesInstanceUID",
        ]

        check_required_columns(df, required_columns)

    anon_patient_dict = {}
    anon_study_dict = {}
    patients = df["PatientID"].dropna().unique()
    for i, patient in tqdm(
        enumerate(patients),
        desc="Anonymizing dataset",
        total=len(patients)
    ):
        anon_patient_dict[patient] = f"sub-{i+1:02d}"
        patient_df = (
            df[df["PatientID"] == patient]
            .copy()
            .sort_values(["StudyDate", "StudyInstanceUID"])
        )


        for j, study in enumerate(patient_df["StudyInstanceUID"].unique()):
            anon_study_dict[study] = f"ses-{j+1:02d}"

    df["AnonPatientID"] = df["PatientID"].apply(
        lambda x: anon_patient_dict[x] if not pd.isna(x) else x
    )

    df["AnonStudyID"] = df["StudyInstanceUID"].apply(
        lambda x: anon_study_dict[x] if not pd.isna(x) else x
    )

    df = (
        df.drop_duplicates(subset="SeriesInstanceUID")
        .sort_values(["AnonPatientID", "AnonStudyID"])
        .reset_index(drop=True)
    )
    return df


def dcm_batch_processor(
    batch: Sequence[Path],
    reorg_dir: Path | str | None = None
):
    """
    Extracts metadata from a sequence of DICOM files and writes
    the file to a new location if a reorganization directory is
    specified.

    Parameters
    ----------
    batch: Sequence[Path]
        A sequence of DICOM instances from which to extract metadata.

    reorg_dir: Path | str | None
        The directory to which files are reorganized if a value
        other than `None` is provided. Defaults to `None`.

    Returns
    -------
    List[Dict[str, str]]
        A list of dictionaries mapping metadata keys to the values
        in a DICOM instance.
    """
    rows = []

    config.settings.reading_validation_mode = config.IGNORE

    for file in batch:
        row = {}

        try:
            dcm = dcmread(file, stop_before_pixels=reorg_dir is None)

            for key in META_KEYS + ["SOPInstanceUID"]:
                # fail on essential meta
                if key in REQUIRED_KEYS:
                    row[key] = str(getattr(dcm, key))

                else:
                    row[key] = str(getattr(dcm, key, None))


        except Exception:
            continue

        if reorg_dir is not None:
            series_dir = Path(reorg_dir) / str(dcm.SeriesInstanceUID)
            series_dir.mkdir(parents=True, exist_ok=True)

            file = series_dir / f"{dcm.SOPInstanceUID}.dcm"
            dcm.save_as(file)

        row["Dicoms"] = file
        rows.append(row)

    return rows


def series_meta(
    patient: Path | str,
    file_extension: Literal["*", "*.dcm"] = "*"
):
    """
    Extracts metadata from each series within a patient directory
    following the organizational scheme of a MIDAS pull.

    Parameters
    ----------
    batch: Path | str
        A sequence of DICOM instances from which to extract metadata.

    file_extension: str
        The assumed file extension used to identify DICOMs. The choices
        are ['*', '*.dcm']. Defaults to '*'.

    Returns
    -------
    List[Dict[str, str]]
        A list of dictionaries mapping metadata keys to the values
        of a DICOM series.
    """
    rows = []

    config.settings.reading_validation_mode = config.IGNORE

    for series in Path(patient).glob("*/*/*/"):
        row = {}

        for file in list(series.glob(file_extension)):
            try:
                dcm = dcmread(file, stop_before_pixels=True)

                for key in META_KEYS + ["SOPInstanceUID"]:
                    # fail on essential meta
                    if key in REQUIRED_KEYS:
                        row[key] = str(getattr(dcm, key))

                    else:
                        row[key] = getattr(dcm, key, None)

                row["Dicoms"] = series
                rows.append(row)

                break

            except Exception:
                continue

    return rows


def create_dicom_dataset(
    dicom_dir: Path | str,
    dataset_csv: Path | str,
    reorg_dir: Path | str | None = None,
    anon: Literal["is_anon", "auto", "deferred"] = "auto",
    batch_size: int = 1000,
    file_extension: Literal["*", "*.dcm"] = "*",
    mode: Literal["arbitrary", "midas"] = "arbitrary",
    cpus: int = 1
):
    """
    Create a DICOM dataset CSV compatible with subsequent `preprocessing`
    scripts. The final CSV provides a series level summary of the location
    of each series alongside metadata extracted from DICOM headers.  If the
    previous organization schems of the dataset does not enforce a DICOM
    series being isolated to a unique directory (instances belonging to
    multiple series must not share the same lowest level directory),
    reorganization must be applied for NIfTI conversion.


    Parameters
    ----------
    dicom_dir: Path | str
        The directory in which the DICOM data is originally stored.

    dataset_csv: Path | str
        The filepath of the output CSV which defines the constructed
        dataset. A corresponding instance level CSV will also be written
        out.

    reorg_dir: Path | str | None
        The directory to which files are reorganized if a value
        other than `None` is provided. Defaults to `None`.

    anon: str
        The anonymization scheme to apply to the completed CSV. Choose
        from:
            'is_anon'
                Assumes the data is already anonymized and uses the
                'PatientID' and 'StudyDate' values.

            'auto'
                Apply automated anonymization to the CSV. This function
                assumes that the 'PatientID' and 'StudyID' tags are
                consistent and correct to derive 'AnonPatientID' = 'sub_{i:02d}'
                and 'AnonStudyID' = 'ses_{i:02d}'.

            'deferred'
                Skip anonymization of the generated CSV. This step will be
                required for subsequent scripts.

    batch_size: int
        The size of the groups of files on which metadata extraction is applied.

    file_extension: str
        The assumed file extension used to identify DICOMs. The choices
        are ['*', '*.dcm']. Defaults to '*'.

    mode: str
        The assumed data orgnaization scheme under `dicom_dir`. The choices
        are ['arbitrary', 'midas']. Defaults to 'arbitrary'.

    cpus: int
        Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing).

    Returns
    -------
    None
        This function does not return anything+, but a CSV is generated to the location
        specified by `dataset_csv`. An instance level CSV is also generated to `str(dataset_csv).replace('.csv', '_instances.csv')`.
        An error file is potentially generated to the same parent directory of these CSVs.
    """
    dicom_dir = Path(dicom_dir).resolve()
    dataset_csv = Path(dataset_csv).resolve()
    cpus = max(cpus, 1)
    dataset_csv.parent.mkdir(parents=True, exist_ok=True)
    errorfile = dataset_csv.parent /  f"{str(datetime.datetime.now()).replace(' ', '_')}.txt"

    if mode == "arbitrary":
        instance_csv = Path(str(dataset_csv).replace(".csv", "_instances.csv"))

        with (
            tqdm(total=0, desc="Constructing DICOM dataset", dynamic_ncols=True) as pbar,
            ProcessPoolExecutor(cpus) as executor,
            instance_csv.open("w", newline="") as instance_csv_io
        ):
            writer = csv.DictWriter(
                instance_csv_io,
                fieldnames=META_KEYS + ["SOPInstanceUID", "Dicoms"],
            )
            writer.writeheader()

            futures = set()
            future_map = {}

            for batch in cglob(
                root=dicom_dir,
                pattern=file_extension,
                batch_size=batch_size,
                queue_size=2 * cpus
            ):
                future = executor.submit(dcm_batch_processor, batch, reorg_dir)
                futures.add(future)
                future_map[future] = batch


                pbar.total += len(batch)
                pbar.refresh()

                completed_futures = set()

                for future in futures:
                    if future.done():
                        try:
                            rows = future.result()

                            for row in rows:
                                writer.writerow(row)

                        except Exception as error:
                            update_errorfile(
                                func_name="preprocessing.data.datasets.dcm_batch_processor",
                                kwargs={"batch": future_map[future]},
                                errorfile=errorfile,
                                error=error
                            )

                            completed_futures.add(future)
                            pbar.update(len(batch))
                            continue

                        completed_futures.add(future)
                        pbar.update(len(batch))

                futures.difference_update(completed_futures)


            for future in as_completed(futures):
                try:
                    rows = future.result()

                    for row in rows:
                        writer.writerow(row)

                except Exception as error:
                    update_errorfile(
                        func_name="preprocessing.data.datasets.dcm_batch_processor",
                        kwargs={"batch": future_map[future]},
                        errorfile=errorfile,
                        error=error
                    )

                    pbar.update(len(future_map[future]))
                    continue

                pbar.update(len(future_map[future]))

        print(f"Dataset of DICOM instances saved to {instance_csv}")

        instance_df = pd.read_csv(instance_csv, dtype=str)

        df = (
            instance_df
            .drop_duplicates(subset=["SeriesInstanceUID"])
            .drop(columns=["SOPInstanceUID"])
        )

        df["Dicoms"] = df["Dicoms"].apply(lambda x: Path(x).parent)


    elif mode == "midas":
        patients = list(dicom_dir.glob("*/"))

        dicts = []

        with ProcessPoolExecutor(cpus) as executor:
            futures = [executor.submit(series_meta, patient, file_extension) for patient in patients]
            for future in tqdm(as_completed(futures), desc="Extracting metadata", total=len(patients)):
                dicts += future.result()

        df = pd.DataFrame(dicts)


    if anon == "is_anon":
        df["AnonPatientID"] = df["PatientID"]
        df["AnonStudyID"] = df["StudyDate"]
        print("Anonymization completed")

    elif anon == "auto":
        df = anonymize_df(df)
        print("Anonymization completed")

    else:
        print(
            "Anonymization has been skipped. Add the 'AnonPatientID' and "
            "'AnonStudyID' manually before running subsequent commands."
        )

    df.to_csv(dataset_csv, index=False)
    print(f"Dataset written to {dataset_csv}")

    if Path(errorfile).exists():
        print(f"Errors were encountered while running this script. Refer to {errorfile}")


def nifti_batch_processor(
    batch: Sequence[Path],
    file_pattern: str,
    normalized_descriptions: bool = True,
    seg_series: str | None = None,
    seg_target: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Record the locations of NIfTI following a consistent naming convention. For
    subsequent `preprocessing` scripts, DICOM metadata is simulated for
    compatibility. Specifically, ['PatientID', 'StudyDate', 'SeriesInstanceUID',
    'SeriesDescription'] (and optionally 'NormalizedSeriesDescription') are
    inferred or randomly generated.

    Parameters
    ----------
    batch: Path | str
        A sequence of DICOM instances from which to extract metadata.

    file_pattern: str
        The file naming convention (without file extensions) of NIfTIs within a
        dataset. Variable names are encoded using '{}' (e.g. `file_pattern` =
        '{patient}_{study}_{series}' would find values for the `patient`, `study`,
        and `series` variables). The `patient`, `study`, and `series` variables
        must be defined.

    normalized_descriptions: bool
        Whether series descriptions are assumed to be normalized. If `True`, the
        'NormalizedSeriesDescription' column will be populated. Defaults to `True`.

    seg_series: str | None
        The series description of segmentations within the dataset, assuming a
        consistent value is present. Must also specify `seg_target` to handle
        segmentations properly. Defaults to `None`.

    seg_target: str | None
        The series description of the series from which segmentations are derived,
        assuming a consistent value is present. Must also specify `seg_series` to
        handle segmentations properly. Defaults to `None`.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A pair of DataFrames containing the locations of considered files. The first
        contains files accepted into the dataset and their simulated metadata, while
        the second records files that encountered an exception and what exception was
        raised.
    """
    accepted_rows = []
    rejected_rows = []

    for file in batch:
        base_name = file.name.split(".")[0]

        try:
            name_map = parse_string(base_name, file_pattern)
            patient, study, series = [
                name_map[k] for k in ["patient", "study", "series"]
            ]

        except Exception as error:
            rejected_rows.append({
                "Nifti": str(file),
                "RejectionError": str(error)
            })

            continue

        if series == seg_series:
            continue

        row = {
            "PatientID": patient,
            "StudyDate": study,
            "SeriesInstanceUID": generate_uid(),
            "SeriesDescription": series,
        }

        if normalized_descriptions:
            row["NormalizedSeriesDescription"] = series

        row["Nifti"] = str(file)

        if series == seg_target and seg_series is not None:
            seg = file.parent / file.name.replace(series, seg_series)
            if seg.exists():
                row["Seg"] = str(seg)

        accepted_rows.append(row)


    return (pd.DataFrame(accepted_rows), pd.DataFrame(rejected_rows))



def create_nifti_dataset(
    nifti_dir: Path | str,
    dataset_csv: Path | str,
    anon: Literal["is_anon", "auto", "deferred"] = "auto",
    batch_size: int = 20,
    batch_processor: Callable = nifti_batch_processor,
    processor_kwargs: Dict[str, Any] = {
        "file_pattern": "{patient}_{study}_{series}",
        "seg_series": "seg",
        "seg_target": "T1Post"
    },
    cpus: int = 1
):
    """
    Create a NIfTI dataset CSV compatible with subsequent `preprocessing`
    scripts. The final CSV provides a series level summary of the location
    of each series alongside metadata generated to simulate DICOM headers.
    Specifically, ['PatientID', 'StudyDate', 'SeriesInstanceUID',
    'SeriesDescription', 'StudyInstanceUID'] (and optionally
    'NormalizedSeriesDescription') are inferred or randomly generated.

    Parameters
    ----------
    nifti_dir: Path | str
        The directory in which the DICOM data is originally stored.

    dataset_csv: Path | str
        The filepath of the output CSV which defines the constructed
        dataset.

    anon: str
        The anonymization scheme to apply to the completed CSV. Choose
        from:
            'is_anon'
                Assumes the data is already anonymized and uses the
                'PatientID' and 'StudyDate' values.

            'auto'
                Apply automated anonymization to the CSV. This function
                assumes that the 'PatientID' and 'StudyID' tags are
                consistent and correct to derive 'AnonPatientID' = 'sub_{i:02d}'
                and 'AnonStudyID' = 'ses_{i:02d}'.

            'deferred'
                Skip anonymization of the generated CSV. This step will be
                required for subsequent scripts.

    batch_size: int
        The size of the groups of files on which metadata extraction is applied.

    batch_processor: Callable
        The function used to infer or generate metadata from a batch of NIfTI files.
        It must accept the key word argument 'batch' and derive ['PatientID',
        'StudyDate', 'SeriesInstanceUID', 'SeriesDescription'] (and optionally
        'NormalizedSeriesDescription') for each file, returning a tuple containing a
        DatFrame of accepted NIfTIs with these metadata and the file location recorded
        under 'Nifti' and a second DataFrame with no constraints. This function can be
        passed any other desired arguments provided they are specified in
        `processor_kwargs`.

    processor_kwargs: Dict[str, Any]
        Additional arguments to be passed to `batch_processor` as key word arguments,
        if applicable.

    cpus: int
        Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing).

    Returns
    -------
    None
        This function does not return anything+, but a CSV is generated to the location
        specified by `dataset_csv`. An error file is potentially generated to the
        same parent directory of the CSV.
    """
    nifti_dir = Path(nifti_dir).resolve()
    dataset_csv = Path(dataset_csv).resolve()
    cpus = max(cpus, 1)
    dataset_csv.parent.mkdir(parents=True, exist_ok=True)
    rejection_csv = str(dataset_csv).replace(".csv", "_rejections.csv")
    errorfile = dataset_csv.parent /  f"{str(datetime.datetime.now()).replace(' ', '_')}.txt"

    accepted_dfs = []
    rejected_dfs = []


    with tqdm(
        total=0, desc="Constructing NIfTI dataset", dynamic_ncols=True
    ) as pbar, ProcessPoolExecutor(cpus) as executor:
        futures = set()
        future_map = {}

        for batch in cglob(
            root=nifti_dir,
            pattern="*.nii*",
            batch_size=batch_size,
            queue_size=2 * cpus
        ):
            kwargs = {"batch": batch, **processor_kwargs}
            future = executor.submit(batch_processor, **kwargs)
            futures.add(future)
            future_map[future] = kwargs


            pbar.total += 1
            pbar.refresh()

            completed_futures = set()

            for future in futures:
                if future.done():
                    try:
                        accepted_df, rejected_df = future.result()

                    except Exception as error:
                        update_errorfile(
                            func_name=f"{batch_processor.__module__}.{batch_processor.__name__}",
                            kwargs=future_map[future],
                            errorfile=errorfile,
                            error=error
                        )

                        completed_futures.add(future)
                        pbar.update(1)
                        continue

                    accepted_dfs.append(accepted_df)
                    rejected_dfs.append(rejected_df)

                    completed_futures.add(future)
                    pbar.update(1)

            futures.difference_update(completed_futures)


        for future in as_completed(futures):
            try:
                accepted_df, rejected_df = future.result()

            except Exception as error:
                update_errorfile(
                    func_name=f"{batch_processor.__module__}.{batch_processor.__name__}",
                    kwargs=future_map[future],
                    errorfile=errorfile,
                    error=error
                )

                pbar.update(1)
                continue

            accepted_dfs.append(accepted_df)
            rejected_dfs.append(rejected_df)
            pbar.update(1)

    df = pd.concat(accepted_dfs)
    discarded_df = pd.concat(rejected_dfs)

    if df.empty:
        print("No matching files have been found and a dataset could not be created")

        return

    study_uid_map = {}

    for patient_id in df["PatientID"].unique():
        patient_df = df[df["PatientID"] == patient_id].copy()

        for study in patient_df["StudyDate"].unique():
            study_uid_map[(patient_id, study)] = generate_uid()

    df["StudyInstanceUID"] = df.apply(
        lambda x: study_uid_map[(x["PatientID"], x["StudyDate"])],
        axis=1
    )

    if anon == "is_anon":
        df["AnonPatientID"] = df["PatientID"]
        df["AnonStudyID"] = df["StudyDate"]
        print("Anonymization completed")

    elif anon == "auto":
        df = anonymize_df(df)
        print("Anonymization completed")

    else:
        print(
            "Anonymization has been skipped. Add the 'AnonPatientID' and "
            "'AnonPatientID' manually before running subsequent commands."
        )

    columns = [key for key in META_KEYS if key in df.keys()]
    columns += [key for key in df.keys() if key not in columns]

    df = df[columns]

    df.to_csv(dataset_csv, index=False)
    print(f"Dataset written to {dataset_csv}")

    if not discarded_df.empty:
        discarded_df.to_csv(rejection_csv, index=False)
        print(
            "Additional NIfTI files were found, but did not follow expected format. "
            f"Rejected files are recorded in {rejection_csv}"
        )

    if Path(errorfile).exists():
        print(f"Errors were encountered while running this script. Refer to {errorfile}")


__all__ = [
    "anonymize_df",
    "create_dicom_dataset",
    "create_nifti_dataset",
]

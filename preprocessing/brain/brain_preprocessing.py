### imports
import os
import shutil
import pandas as pd
import numpy as np
import nibabel as nib
import multiprocessing
import json

from nipype.interfaces.ants import N4BiasFieldCorrection
from pathlib import Path
from subprocess import run
from tqdm import tqdm
from preprocessing.utils import source_external_software, check_required_columns


def copy_metadata(row: dict, preprocessing_args: dict) -> None:
    original_metafile = row["nifti"].replace(".nii.gz", ".json")
    if Path(original_metafile).exists():
        with open(original_metafile, "r") as json_file:
            data = json.load(json_file)
        meta_dict = {
            "source_file": row["nifti"],
            "original_metafile": data,
            "preprocessing_args": preprocessing_args,
        }
        preprocessed_metafile = row[preprocessing_args["pipeline_key"]].replace(
            ".nii.gz", ".json"
        )
        with open(preprocessed_metafile, "w") as json_file:
            json.dump(
                meta_dict, json_file, sort_keys=True, indent=2, separators=(",", ": ")
            )
    else:
        error = f"{original_metafile} does not exist. New metafile will not be created"
        print(error)
        e = open(f"{preprocessing_args['preprocessed_dir']}/errors.txt", "a")
        e.write(f"{error}\n")

    if "seg" in row:
        original_metafile = row["seg"].replace(".nii.gz", ".json")
        if Path(original_metafile).exists():
            with open(original_metafile, "r") as json_file:
                data = json.load(json_file)
            meta_dict = {
                "source_file": row["nifti"],
                "original_metafile": data,
                "preprocessing_args": preprocessing_args,
            }
            preprocessed_metafile = row[
                f"{preprocessing_args['pipeline_key']}_seg"
            ].replace(".nii.gz", ".json")
            with open(preprocessed_metafile, "w") as json_file:
                json.dump(
                    meta_dict,
                    json_file,
                    sort_keys=True,
                    indent=2,
                    separators=(",", ": "),
                )
        else:
            error = (
                f"{original_metafile} does not exist. New metafile will not be created"
            )
            print(error)
            e = open(f"{preprocessing_args['preprocessed_dir']}/errors.txt", "a")
            e.write(f"{error}\n")


def preprocess_study(
    study_df: pd.DataFrame,
    preprocessed_dir: Path,
    pipeline_key: str,
    registration_key: str = "T1Post",
    registration_target: str | None = None,
    orientation: str = "RAI",
    spacing: str = "1,1,1",
    skullstrip: bool = True,
    verbose: bool = False,
    source_software: bool = True,
    check_columns: bool = True,
) -> pd.DataFrame:
    if source_software:
        source_external_software()

    if check_columns:
        required_columns = [
            "nifti",
            "Anon_PatientID",
            "Anon_StudyID",
            "StudyInstanceUID",
            "NormalizedSeriesDescription",
            "SeriesType",
        ]
        optional_columns = ["seg"]

        check_required_columns(study_df, required_columns, optional_columns)

    def registration_sort(series):
        return (series != registration_key).astype(int)

    filtered_df = (
        study_df.copy()
        .dropna(subset="NormalizedSeriesDescription")
        .sort_values(["NormalizedSeriesDescription"], key=registration_sort)
    )
    if filtered_df.empty:
        return study_df

    anon_patientID = filtered_df.loc[filtered_df.index[0], "Anon_PatientID"]
    anon_studyID = filtered_df.loc[filtered_df.index[0], "Anon_StudyID"]

    rows = filtered_df.to_dict("records")
    n = len(rows)

    # must enforce one normalizedseries description per study
    ### copy files to new location
    for i in range(n):
        output_dir = (
            preprocessed_dir / anon_patientID / anon_studyID / rows[i]["SeriesType"]
        )
        os.makedirs(output_dir, exist_ok=True)

        input_file = rows[i]["nifti"]
        preprocessed_file = output_dir / os.path.basename(input_file)
        shutil.copy(input_file, preprocessed_file)

        if os.path.exists(preprocessed_file):
            rows[i][pipeline_key] = str(preprocessed_file)
        else:
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy(input_file, preprocessed_file)
            if os.path.exists(preprocessed_file):
                rows[i][pipeline_key] = str(preprocessed_file)
            else:
                error = f"Could not create {preprocessed_file}"
                print(error)
                e = open(f"{preprocessed_dir}/errors.txt", "a")
                e.write(f"{error}\n")
                return study_df

        if "seg" in rows[i]:
            input_file = rows[i]["seg"]
            preprocessed_seg = output_dir / os.path.basename(input_file)
            shutil.copy(input_file, preprocessed_seg)

            if os.path.exists(preprocessed_seg):
                rows[i][f"{pipeline_key}_seg"] = str(preprocessed_seg)
            else:
                os.makedirs(output_dir, exist_ok=True)
                shutil.copy(input_file, preprocessed_seg)
                if os.path.exists(preprocessed_file):
                    rows[i][f"{pipeline_key}_seg"] = str(preprocessed_file)
                else:
                    error = f"Could not create {preprocessed_seg}"
                    print(error)
                    e = open(f"{preprocessed_dir}/errors.txt", "a")
                    e.write(f"{error}\n")
                    return study_df

    ### Orientation
    for i in range(n):
        preprocessed_file = rows[i][pipeline_key]

        command = (
            f"Slicer --launch OrientScalarVolume "
            f"{preprocessed_file} {preprocessed_file} -o {orientation}"
        )
        if verbose:
            print(command)
        try:
            run(command.split(" "), capture_output=not verbose).check_returncode()

        except Exception as error:
            print(error)
            e = open(f"{preprocessed_dir}/errors.txt", "a")
            e.write(f"{error}\n")
            return study_df

        if "seg" in rows[i]:
            preprocessed_seg = rows[i][f"{pipeline_key}_seg"]

            command = (
                f"Slicer --launch OrientScalarVolume "
                f"{preprocessed_seg} {preprocessed_seg} -o {orientation}"
            )
            if verbose:
                print(command)
            try:
                run(command.split(" "), capture_output=not verbose).check_returncode()

            except Exception as error:
                print(error)
                e = open(f"{preprocessed_dir}/errors.txt", "a")
                e.write(f"{error}\n")
                return study_df

    ### Spacing
    for i in range(n):
        preprocessed_file = rows[i][pipeline_key]
        command = (
            f"Slicer --launch ResampleScalarVolume "
            f"{preprocessed_file} {preprocessed_file} -i bspline -s {spacing}"
        )
        if verbose:
            print(command)
        try:
            run(command.split(" "), capture_output=not verbose).check_returncode()

        except Exception as error:
            print(error)
            e = open(f"{preprocessed_dir}/errors.txt", "a")
            e.write(f"{error}\n")
            return study_df

        if "seg" in rows[i]:
            preprocessed_seg = rows[i][f"{pipeline_key}_seg"]
            command = (
                f"Slicer --launch ResampleScalarVolume "
                f"{preprocessed_seg} {preprocessed_seg} -i nearestNeighbor -s {spacing}"
            )
            if verbose:
                print(command)
            try:
                run(command.split(" "), capture_output=not verbose).check_returncode()

            except Exception as error:
                print(error)
                e = open(f"{preprocessed_dir}/errors.txt", "a")
                e.write(f"{error}\n")
                return study_df

    ### Loose Skullstrip
    for i in range(n):
        preprocessed_file = rows[i][pipeline_key]
        SS_file = preprocessed_file.replace(".nii.gz", "_SS.nii.gz")
        SS_mask = preprocessed_file.replace(".nii.gz", "_SS_mask.nii.gz")

        command = f"mri_synthstrip -i {preprocessed_file} -o {SS_file} -m {SS_mask}"
        if verbose:
            print(command)
        try:
            run(command.split(" "), capture_output=not verbose).check_returncode()

        except Exception as error:
            print(error)
            e = open(f"{preprocessed_dir}/errors.txt", "a")
            e.write(f"{error}\n")
            return study_df

    if registration_target is None:
        main_SS_file = rows[0][pipeline_key].replace(".nii.gz", "_SS.nii.gz")
        main_SS_mask_file = rows[0][pipeline_key].replace(".nii.gz", "_SS_mask.nii.gz")
        main_SS_mask_array = np.round(nib.load(main_SS_mask_file).get_fdata())
    else:
        main_SS_file = registration_target.replace(".nii.gz", "_SS.nii.gz")
        main_SS_mask_file = registration_target.replace(".nii.gz", "_SS_mask.nii.gz")
        main_SS_mask_array = np.round(nib.load(main_SS_mask_file).get_fdata())

    ### Register based on loose skullstrip
    fixed_image_path = main_SS_file
    if registration_target is None:
        if n > 1:
            for i in range(1, n):
                preprocessed_file = rows[i][pipeline_key]
                moving_image_path = preprocessed_file.replace(".nii.gz", "_SS.nii.gz")
                transform_outfile = preprocessed_file.replace(
                    ".nii.gz", "_transform.tfm"
                )
                sampling_percentage = 0.002
                x, y, z = main_SS_mask_array.shape

                command = (
                    f"Slicer "
                    f"--launch BRAINSFit --fixedVolume {fixed_image_path} "
                    f"--movingVolume {moving_image_path} "
                    "--transformType Rigid,ScaleVersor3D,ScaleSkewVersor3D,Affine "
                    "--initializeTransformMode useGeometryAlign "
                    "--interpolationMode BSpline "
                    f"--outputTransform {transform_outfile} "
                    f"--samplingPercentage {sampling_percentage} "
                )

                if verbose:
                    print(command)
                try:
                    run(
                        command.split(" "), capture_output=not verbose
                    ).check_returncode()

                except Exception as error:
                    print(error)
                    e = open(f"{preprocessed_dir}/errors.txt", "a")
                    e.write(f"{error}\n")
                    return study_df

                command = (
                    f"Slicer --launch ResampleScalarVectorDWIVolume "
                    f"{preprocessed_file} {preprocessed_file} -i bs -f {transform_outfile}"
                )
                if verbose:
                    print(command)
                try:
                    run(
                        command.split(" "), capture_output=not verbose
                    ).check_returncode()

                except Exception as error:
                    print(error)
                    e = open(f"{preprocessed_dir}/errors.txt", "a")
                    e.write(f"{error}\n")
                    return study_df

                command = (
                    f"Slicer --launch ResampleScalarVectorDWIVolume "
                    f"{preprocessed_file} {preprocessed_file} -i bs -R {main_SS_file} -z {x},{y},{z}"
                )
                if verbose:
                    print(command)
                try:
                    run(
                        command.split(" "), capture_output=not verbose
                    ).check_returncode()

                except Exception as error:
                    print(error)
                    e = open(f"{preprocessed_dir}/errors.txt", "a")
                    e.write(f"{error}\n")
                    return study_df

                if "seg" in rows[i]:
                    preprocessed_seg = rows[i][f"{pipeline_key}_seg"]

                    command = (
                        f"Slicer --launch ResampleScalarVectorDWIVolume "
                        f"{preprocessed_seg} {preprocessed_seg} -i nn -f {transform_outfile}"
                    )
                    if verbose:
                        print(command)
                    try:
                        run(
                            command.split(" "), capture_output=not verbose
                        ).check_returncode()

                    except Exception as error:
                        print(error)
                        e = open(f"{preprocessed_dir}/errors.txt", "a")
                        e.write(f"{error}\n")
                        return study_df

                    command = (
                        f"Slicer --launch ResampleScalarVectorDWIVolume "
                        f"{preprocessed_seg} {preprocessed_seg} -i nn -R {main_SS_file} -z {x},{y},{z}"
                    )
                    if verbose:
                        print(command)
                    try:
                        run(
                            command.split(" "), capture_output=not verbose
                        ).check_returncode()

                    except Exception as error:
                        print(error)
                        e = open(f"{preprocessed_dir}/errors.txt", "a")
                        e.write(f"{error}\n")
                        return study_df

    else:
        for i in range(n):
            preprocessed_file = rows[i][pipeline_key]
            moving_image_path = preprocessed_file.replace(".nii.gz", "_SS.nii.gz")
            transform_outfile = preprocessed_file.replace(".nii.gz", "_transform.tfm")
            sampling_percentage = 0.002
            x, y, z = main_SS_mask_array.shape

            command = (
                f"Slicer "
                f"--launch BRAINSFit --fixedVolume {fixed_image_path} "
                f"--movingVolume {moving_image_path} "
                "--transformType Rigid,ScaleVersor3D,ScaleSkewVersor3D,Affine "
                "--initializeTransformMode useGeometryAlign "
                "--interpolationMode BSpline "
                f"--outputTransform {transform_outfile} "
                f"--samplingPercentage {sampling_percentage} "
            )

            if verbose:
                print(command)
            try:
                run(command.split(" "), capture_output=not verbose).check_returncode()

            except Exception as error:
                print(error)
                e = open(f"{preprocessed_dir}/errors.txt", "a")
                e.write(f"{error}\n")
                return study_df

            command = (
                f"Slicer --launch ResampleScalarVectorDWIVolume "
                f"{preprocessed_file} {preprocessed_file} -i bs -f {transform_outfile}"
            )
            if verbose:
                print(command)
            try:
                run(command.split(" "), capture_output=not verbose).check_returncode()

            except Exception as error:
                print(error)
                e = open(f"{preprocessed_dir}/errors.txt", "a")
                e.write(f"{error}\n")
                return study_df

            command = (
                f"Slicer --launch ResampleScalarVectorDWIVolume "
                f"{preprocessed_file} {preprocessed_file} -i bs -R {main_SS_file} -z {x},{y},{z}"
            )
            if verbose:
                print(command)
            try:
                run(command.split(" "), capture_output=not verbose).check_returncode()

            except Exception as error:
                print(error)
                e = open(f"{preprocessed_dir}/errors.txt", "a")
                e.write(f"{error}\n")
                return study_df

            if "seg" in rows[i]:
                preprocessed_seg = rows[i][f"{pipeline_key}_seg"]

                command = (
                    f"Slicer --launch ResampleScalarVectorDWIVolume "
                    f"{preprocessed_seg} {preprocessed_seg} -i nn -f {transform_outfile}"
                )
                if verbose:
                    print(command)
                try:
                    run(
                        command.split(" "), capture_output=not verbose
                    ).check_returncode()

                except Exception as error:
                    print(error)
                    e = open(f"{preprocessed_dir}/errors.txt", "a")
                    e.write(f"{error}\n")
                    return study_df

                command = (
                    f"Slicer --launch ResampleScalarVectorDWIVolume "
                    f"{preprocessed_seg} {preprocessed_seg} -i nn -R {main_SS_file} -z {x},{y},{z}"
                )
                if verbose:
                    print(command)
                try:
                    run(
                        command.split(" "), capture_output=not verbose
                    ).check_returncode()

                except Exception as error:
                    print(error)
                    e = open(f"{preprocessed_dir}/errors.txt", "a")
                    e.write(f"{error}\n")
                    return study_df

    ### appy final skullmask if skullstripping
    if skullstrip:
        for i in range(n):
            preprocessed_file = rows[i][pipeline_key]
            nifti = nib.load(preprocessed_file)
            array = nifti.get_fdata()

            array = array * main_SS_mask_array

            output_nifti = nib.Nifti1Image(
                array, affine=nifti.affine, header=nifti.header
            )
            nib.save(output_nifti, preprocessed_file)

            if "seg" in rows[i]:
                preprocessed_seg = rows[i][f"{pipeline_key}_seg"]
                nifti = nib.load(preprocessed_seg)
                array = nifti.get_fdata()

                array = array * main_SS_mask_array

                output_nifti = nib.Nifti1Image(
                    array, affine=nifti.affine, header=nifti.header
                )
                nib.save(output_nifti, preprocessed_seg)

    ### Bias correction
    for i in range(n):
        preprocessed_file = rows[i][pipeline_key]

        n4 = N4BiasFieldCorrection()
        n4.inputs.input_image = preprocessed_file
        n4.inputs.n_iterations = [20, 20, 10, 5]
        n4.inputs.output_image = preprocessed_file
        n4.run()

    ### Normalization
    for i in range(n):
        preprocessed_file = rows[i][pipeline_key]

        nifti = nib.load(preprocessed_file)
        array = nifti.get_fdata()
        masked_input_array = np.ma.masked_where(main_SS_mask_array == 0, array)
        mean = np.ma.mean(masked_input_array)
        std = np.ma.std(masked_input_array)
        array = (array - mean) / std

        output_nifti = nib.Nifti1Image(array, affine=nifti.affine, header=nifti.header)
        nib.save(output_nifti, preprocessed_file)

    ### copy metadata
    preprocessing_args = {
        "preprocessed_dir": str(preprocessed_dir),
        "pipeline_key": pipeline_key,
        "registration_target": registration_target,
        "orientation": orientation,
        "spacing": spacing,
        "skullstrip": skullstrip,
    }

    for row in rows:
        copy_metadata(row, preprocessing_args)

    preprocessed_df = pd.DataFrame(rows)
    out_df = pd.merge(study_df, preprocessed_df, "outer")

    return out_df


def preprocess_patient(
    patient_df: pd.DataFrame,
    preprocessed_dir: Path,
    pipeline_key: str = "preprocessed",
    registration_key: str = "T1Post",
    longitudinal_registration: bool = False,
    orientation: str = "RAI",
    spacing: str = "1,1,1",
    skullstrip: bool = True,
    verbose: bool = False,
    source_software: bool = True,
    check_columns: bool = True,
):
    if source_software:
        source_external_software()

    if check_columns:
        required_columns = [
            "nifti",
            "Anon_PatientID",
            "Anon_StudyID",
            "StudyInstanceUID",
            "NormalizedSeriesDescription",
            "SeriesType",
        ]
        optional_columns = ["seg"]

        check_required_columns(patient_df, required_columns, optional_columns)

    study_uids = patient_df["StudyInstanceUID"].unique()

    preprocessed_dfs = []

    study_df = patient_df[patient_df["StudyInstanceUID"] == study_uids[0]].copy()

    preprocessed_dfs.append(
        preprocess_study(
            study_df=study_df,
            preprocessed_dir=preprocessed_dir,
            pipeline_key=pipeline_key,
            registration_key=registration_key,
            registration_target=None,
            orientation=orientation,
            spacing=spacing,
            skullstrip=skullstrip,
            verbose=verbose,
            source_software=False,
            check_columns=False,
        )
    )

    if len(study_uids) > 1:
        if longitudinal_registration:
            # TODO change registration target by a series description key
            registration_target = preprocessed_dfs[0][pipeline_key][0]

            for study_uid in study_uids[1:]:
                study_df = patient_df[
                    patient_df["StudyInstanceUID"] == study_uid
                ].copy()

                preprocessed_dfs.append(
                    preprocess_study(
                        study_df=study_df,
                        preprocessed_dir=preprocessed_dir,
                        pipeline_key=pipeline_key,
                        registration_key=registration_key,
                        registration_target=registration_target,
                        orientation=orientation,
                        spacing=spacing,
                        skullstrip=skullstrip,
                        verbose=verbose,
                        source_software=False,
                        check_columns=False,
                    )
                )

        else:
            for study_uid in study_uids[1:]:
                study_df = patient_df[
                    patient_df["StudyInstanceUID"] == study_uid
                ].copy()

                preprocessed_dfs.append(
                    preprocess_study(
                        study_df=study_df,
                        preprocessed_dir=preprocessed_dir,
                        pipeline_key=pipeline_key,
                        registration_key=registration_key,
                        registration_target=None,
                        orientation=orientation,
                        spacing=spacing,
                        skullstrip=skullstrip,
                        verbose=verbose,
                        source_software=False,
                        check_columns=False,
                    )
                )

    # clear extra files
    anon_patientID = patient_df.loc[patient_df.index[0], "Anon_PatientID"]
    patient_dir = preprocessed_dir / anon_patientID

    out_df = pd.concat(preprocessed_dfs, ignore_index=True)

    extra_files = (
        list(patient_dir.glob("**/*SS.nii.gz"))
        + list(patient_dir.glob("**/*SS_mask.nii.gz"))
        + list(patient_dir.glob("**/*.tfm"))
        + list(patient_dir.glob("**/*.h5"))
    )

    print("......Clearing unnecessary files......")
    for file in extra_files:
        os.remove(file)
    print(f"Finished preprocessing {anon_patientID}:")
    print(out_df)
    return out_df


def preprocess_patient_star(args):
    return preprocess_patient(*args)


def preprocess_from_csv(
    csv: Path | str,
    preprocessed_dir: Path | str,
    pipeline_key: str = "preprocessed",
    registration_key: str = "T1Post",
    longitudinal_registration: bool = False,
    orientation: str = "RAI",
    spacing: str = "1,1,1",
    skullstrip: bool = True,
    cpus: int = 0,
    verbose: bool = False,
) -> pd.DataFrame:
    source_external_software()

    preprocessed_dir = Path(preprocessed_dir)

    df = pd.read_csv(csv)

    required_columns = [
        "nifti",
        "Anon_PatientID",
        "Anon_StudyID",
        "StudyInstanceUID",
        "NormalizedSeriesDescription",
        "SeriesType",
    ]
    optional_columns = ["seg"]

    check_required_columns(df, required_columns, optional_columns)

    filtered_df = df.copy().dropna(subset="nifti")
    patients = filtered_df["Anon_PatientID"].unique()

    if cpus == 0:
        outputs = [
            preprocess_patient(
                filtered_df[filtered_df["Anon_PatientID"] == patient].copy(),
                preprocessed_dir,
                pipeline_key,
                registration_key,
                longitudinal_registration,
                orientation,
                spacing,
                skullstrip,
                verbose,
                False,
                False,
            )
            for patient in tqdm(patients, desc="Preprocessing patients")
        ]

    else:
        inputs = [
            [
                filtered_df[filtered_df["Anon_PatientID"] == patient].copy(),
                preprocessed_dir,
                pipeline_key,
                registration_key,
                longitudinal_registration,
                orientation,
                spacing,
                skullstrip,
                verbose,
                False,
                False,
            ]
            for patient in patients
        ]

        with multiprocessing.Pool(cpus) as pool:
            outputs = list(
                tqdm(
                    pool.imap(preprocess_patient_star, inputs),
                    total=len(patients),
                    desc="Preprocessing patients",
                )
            )

    preprocessed_df = pd.concat(outputs)
    df = pd.merge(df, preprocessed_df, how="outer")
    df = df.sort_values(["Anon_PatientID", "Anon_StudyID"]).reset_index(drop=True)
    df.to_csv(csv, index=False)
    return df

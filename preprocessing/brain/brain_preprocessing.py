### imports
import os
import shutil
import pandas as pd
import numpy as np
import nibabel as nib
import multiprocessing
import json

from nipype.interfaces.ants import N4BiasFieldCorrection
from typing import Sequence
from pathlib import Path
from subprocess import run

slicer_env = {
    "PATH": "/usr/pubsw/packages/slicer/Slicer-5.2.2-linux-amd64/:" + os.environ["PATH"]
}


def copy_metadata(row: dict, preprocessing_args: dict) -> None:
    original_metafile = row["nifti"].replace(".nii.gz", ".json")
    with open(original_metadata, "r") as json_file:
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


def preprocess_study(
    study_df: pd.DataFrame,
    preprocessed_dir: Path,
    pipeline_key: str,
    registration_target: str | None = None,
    orientation: str = "RAI",
    spacing: str = "1,1,1",
    skullstrip: bool = True,
) -> pd.DataFrame:
    anon_patientID = study_df["Anon_PatientID"][0]
    anon_studyID = study_df["Anon_StudyID"][0]

    filtered_df = (
        study_df.copy()
        .dropna(subset="NormalizedSeriesDescription")
        .sort_values(["seg"])  # use the segmentation case for registration
    )
    if filtered_df.empty():
        return study_df

    rows = filtered_df.to_dict("records")
    n = len(rows)

    # must enforce one normalizedseries description per study
    ### copy files to new location
    for i in range(n):
        output_dir = (
            preprocessed_dir / anon_patientID / anon_studyID / rows[i]["SeriesType"]
        )
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

    ### Orientation
    for i in range(n):
        preprocessed_file = rows[i][pipeline_key]

        command = (
            f"Slicer --launch OrientScalarVolume "
            f"{preprocessed_file} {preprocessed_file} -o {orientation}"
        )
        print(command)
        try:
            run(command.split(" "), env=slicer_env).check_returncode()

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
        print(command)
        try:
            run(command.split(" "), env=slicer_env).check_returncode()

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

        command = f"mri_synthstrip -i {input_file} -o {SS_file} -m {SS_mask}"
        print(command)
        try:
            run(command.split(" "), env=slicer_env).check_returncode()

        except Exception as error:
            print(error)
            e = open(f"{preprocessed_dir}/errors.txt", "a")
            e.write(f"{error}\n")
            return study_df

    if registration_target is None:
        main_SS_mask_file = rows[i][pipeline_key].replace(".nii.gz", "_SS.nii.gz")
        main_SS_mask_array = np.round(nib.load(main_SS_mask_file).get_fdata())
    else:
        main_SS_mask_file = registration_target.replace(".nii.gz", "_SS.nii.gz")
        main_SS_mask_array = np.round(nib.load(main_SS_mask_file).get_fdata())

    ### Register based on Loose skullstrip if i>0
    fixed_image_path = main_SS_mask_file
    if registration_target is None:
        if n > 1:
            for i in range(1, n):
                preprocessed_file = rows[i][pipeline_key]
                moving_image_path = preprocessed_file.replace(".nii.gz", "_SS.nii.gz")
                transform_outfile = preprocessed_file.replace(
                    ".nii.gz", "_transform.tfm"
                )
                sampling_percentage = 0.002

                command = (
                    f"Slicer "
                    f"--launch BRAINSFit --fixedVolume {fixed_image_path} "
                    f"--movingVolume {moving_image_path} "
                    "--transformType Rigid,ScaleVersor3D,ScaleSkewVersor3D,Affine "
                    "--initializeTransformMode useGeometryAlign "
                    "--interpolationMode BSpline "
                    f"--outputTransform {transform_outfile} "
                    f"--samplingPercentage {sampling_percentage}"
                )

                print(command)
                try:
                    run(command.split(" "), env=slicer_env).check_returncode()

                except Exception as error:
                    print(error)
                    e = open(f"{preprocessed_dir}/errors.txt", "a")
                    e.write(f"{error}\n")
                    return study_df
                
                command = (
                    f"{slicer_dir} --launch ResampleScalarVectorDWIVolume "
                    f"{preprocessed_file} {preprocessed_file} -i bs -f {transform_outfile}"
                )
                print(command)
                try:
                    run(command.split(" "), env=slicer_env).check_returncode()

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

            command = (
                f"Slicer "
                f"--launch BRAINSFit --fixedVolume {fixed_image_path} "
                f"--movingVolume {moving_image_path} "
                "--transformType Rigid,ScaleVersor3D,ScaleSkewVersor3D,Affine "
                "--initializeTransformMode useGeometryAlign "
                "--interpolationMode BSpline "
                f"--outputTransform {transform_outfile} "
                f"--samplingPercentage {sampling_percentage}"
            )

            print(command)
            try:
                run(command.split(" "), env=slicer_env).check_returncode()

            except Exception as error:
                print(error)
                e = open(f"{preprocessed_dir}/errors.txt", "a")
                e.write(f"{error}\n")
                return study_df

            command = (
                f"{slicer_dir} --launch ResampleScalarVectorDWIVolume "
                f"{preprocessed_file} {preprocessed_file} -i bs -f {transform_outfile}"
            )
            print(command)
            try:
                run(command.split(" "), env=slicer_env).check_returncode()

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

    ### Process segmentation
    input_file = rows[0]["seg"]

    if not np.isnan(input_file):
        output_dir = (
            preprocessed_dir / anon_patientID / anon_studyID / rows[i]["SeriesType"]
        )
        preprocessed_file = output_dir / os.path.basename(input_file)
        shutil.copy(input_file, preprocessed_file)

        if os.path.exists(preprocessed_file):
            rows[i][f"{pipeline_key}_seg"] = str(preprocessed_file)
        else:
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy(input_file, preprocessed_file)
            if os.path.exists(preprocessed_file):
                rows[i][f"{pipeline_key}_seg"] = str(preprocessed_file)
            else:
                error = f"Could not create {preprocessed_file}"
                print(error)
                e = open(f"{preprocessed_dir}/errors.txt", "a")
                e.write(f"{error}\n")
                return study_df

        ### orientation
        command = (
            "Slicer --launch OrientScalarVolume "
            f"{preprocessed_file} {preprocessed_file} -o {orientation}"
        )
        print(command)
        try:
            run(command.split(" "), env=slicer_env).check_returncode()
        except Exception as error:
            print(error)
            e = open(f"{preprocessed_dir}/errors.txt", "a")
            e.write(f"{error}\n")
            return study_df

        ### resample to input file
        reference_file = rows[0][pipeline_key]

        command = (
            "Slicer --launch ResampleScalarVectorDWIVolume "
            f"{preprocessed_file} {preprocessed_file} -i nn -R {reference_file}"
        )
        print(command)
        try:
            run(command.split(" "), env=slicer_env).check_returncode()
        except Exception as error:
            print(error)
            e = open(f"{preprocessed_dir}/errors.txt", "a")
            e.write(f"{error}\n")
            return study_df

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
    pipeline_key: str,
    longitudinal_registration: bool = False,
    orientation: str = "RAI",
    spacing: str = "1,1,1",
    skullstrip: bool = True,
):
    study_uids = patient_df["StudyInstanceUID"].unique()

    preprocessed_dfs = []

    study_df = patient_df[patient_df["StudyInstanceUID"] == study_uids[0]].copy()

    preprocessed_dfs.append(
        preprocess_study(
            study_df=study_df,
            preprocessed_dir=preprocessed_dir,
            pipeline_key=pipeline_key,
            registration_target=None,
            orientation=orientation,
            spacing=spacing,
            skullstrip=skullstrip,
        )
    )
    
    if len(study_uids>1):
        if longitudinal_registration:
            registration_target = preprocessed_dfs[0][pipeline_key][0]

            for study_uid in study_uids[1:]:
                study_df = patient_df[patient_df["StudyInstanceUID"] == study_uid].copy()
                
                preprocessed_dfs.append(
                    preprocess_study(
                        study_df=study_df,
                        preprocessed_dir=preprocessed_dir,
                        pipeline_key=pipeline_key,
                        registration_target=registration_target,
                        orientation=orientation,
                        spacing=spacing,
                        skullstrip=skullstrip,
                    )
                )

        else:
            for study_uid in study_uids[1:]:
                study_df = patient_df[patient_df["StudyInstanceUID"] == study_uid.copy()
                
                preprocessed_dfs.append(
                    preprocess_study(
                        study_df=study_df,
                        preprocessed_dir=preprocessed_dir,
                        pipeline_key=pipeline_key,
                        registration_target=None,
                        orientation=orientation,
                        spacing=spacing,
                        skullstrip=skullstrip,
                    )
                )


    return pd.concat(preprocessed_dfs, ignore_index=True)


def preprocess_from_csv(
    csv: str,
    output_dir: str,
    orientation: str = "RAI",
    spacing: str = "1,1,1",
    skullstrip: bool = True,
    cpus: int = 0,
) -> pd.DataFrame:
    df = pd.read_csv(csv)

    # if cpus == 0:
    #     results = []
    #     for row in rows:
    #         results.append(
    #             preprocess_single_case(row, output_dir, im_keys, seg_keys, skullstrip)
    #         )
    # if cpus > 0:
    #     inputs = [[row, output_dir, im_keys, seg_keys, skullstrip] for row in rows]
    #     with multiprocessing.Pool(cpus) as pool:
    #         results = pool.starmap(preprocess_single_case, inputs)
    #
    # df = pd.DataFrame(results)
    # df.to_csv(csv, index=False)

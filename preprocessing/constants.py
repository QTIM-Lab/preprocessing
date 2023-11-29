META_KEYS = [
    "SeriesInstanceUID",
    "StudyInstanceUID",
    "PatientID",
    "AccessionNumber",
    "StudyDate",
    "StudyDescription",
    "SeriesDescription",
]

ANAT_SERIES_DESCRIPTIONS = [
    "iso3D AX T1 NonContrast",
    "iso3D AX T1 NonContrast RFMT",
    "iso3D AX T1 WithContrast",
    "iso3D AX T1 WithContrast RFMT",
    "iso3D AX T2",
    "iso3D AX T2 RFMT",
    "iso3D AX T2 STIR",
    "iso3D AX T2 STIR RFMT",
]
FUNC_SERIES_DESCRIPTIONS = []
DWI_SERIES_DESCRIPTIONS = []
ALL_SERIES_DESCRIPTIONS = (
    ANAT_SERIES_DESCRIPTIONS + FUNC_SERIES_DESCRIPTIONS + DWI_SERIES_DESCRIPTIONS
)

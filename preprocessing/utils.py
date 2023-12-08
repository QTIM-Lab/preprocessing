import os


def source_external_software():
    os.environ["PATH"] += (
        ":/usr/local/freesurfer/7.3.3/bin"
        ":/usr/pubsw/packages/fsl/6.0.6/bin"
        ":/usr/pubsw/packages/slicer/Slicer-5.2.2-linux-amd64/"
        ":/usr/pubsw/packages/ANTS/2.3.5/bin"
    )
    os.environ["ANTSPATH"] = "/usr/pubsw/packages/ANTS/2.3.5/bin"

    os.environ["FREESURFER_HOME"] = "/usr/local/freesurfer/7.3.3"
    os.system(f"source {os.environ['FREESURFER_HOME']}/SetUpFreeSurfer.sh")

    os.environ["FSLDIR"] = "/usr/pubsw/packages/fsl/6.0.6/"
    os.system(f"source {os.environ['FSLDIR']}/etc/fslconf/fsl.sh")

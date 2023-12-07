import os

# command_env = {
#     "PATH": "/usr/local/freesurfer/7.3.3/bin/:/usr/pubsw/packages/slicer/Slicer-5.2.2-linux-amd64/:"
#     + os.environ["PATH"]
# }


def source_external_software():
    os.environ["FREESURFER_HOME"] = "/usr/local/freesurfer/7.3.3"
    os.environ[
        "PATH"
    ] += ":/usr/local/freesurfer/7.3.3/bin:/usr/pubsw/packages/slicer/Slicer-5.2.2-linux-amd64/"

    os.system(f"source {os.environ['FREESURFER_HOME']}/SetUpFreeSurfer.sh")

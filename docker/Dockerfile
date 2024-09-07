# Using repo root as build context
FROM python:3.11-slim
RUN apt-get update; apt-get install -qy build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl git unzip \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev; rm -rf /root/.cache

ENV PATH=/external_software/bin:$PATH
RUN curl -fLO https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_lnx.zip; unzip dcm2niix_lnx.zip; mkdir -p /external_software/bin; mv dcm2niix /external_software/bin/

ENV PREPROCESSING_MODELS_PATH=/models/
RUN pip install poetry; rm -rf /root/.cache

COPY . /preprocessing/

RUN cd /preprocessing; poetry install --sync --no-root; pip install -e .; rm -rf /root/.cache
RUN preprocessing validate-installation

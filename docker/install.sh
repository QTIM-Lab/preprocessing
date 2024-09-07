#!/bin/bash
REPO_ROOT=$(realpath $(dirname $0)/..)
DOCKER_SCRIPTS=${REPO_ROOT}/docker

docker build -t "qtim/preprocessing:0.1.0" -f ${DOCKER_SCRIPTS}/Dockerfile ${REPO_ROOT}/

export PATH=${DOCKER_SCRIPTS}:${PATH}
chmod +x ${DOCKER_SCRIPTS}/preprocessing-docker

line_in_file() {
    if [ -f "$2" ]; then
        grep -qF "$1" "$2"
    else
	false
    fi
}

if line_in_file "export PATH=${DOCKER_SCRIPTS}:\${PATH}" ${HOME}/.bashrc; then
    echo "preprocessing-docker is already added to \$PATH"
elif line_in_file "export PATH=${DOCKER_SCRIPTS}:\${PATH}" ${HOME}/.zshrc; then
    echo "preprocessing-docker is already added to \$PATH"
else
    if [ -f ~/.bashrc ]; then
        echo "export PATH=${DOCKER_SCRIPTS}:\${PATH}" >> ${HOME}/.bashrc
        echo "Added preprocessing-docker to \$PATH and updated your .bashrc"
    elif [ -f ~/.zshrc ]; then
      	echo "export PATH=${DOCKER_SCRIPTS}:\${PATH}" >> ${HOME}/.zshrc
        echo "Added preprocessing-docker to \$PATH and updated your .zshrc"
    else
	echo "Shell not supported. Add ${DOCKER_SCRIPTS} to \$PATH manually"
    fi
fi

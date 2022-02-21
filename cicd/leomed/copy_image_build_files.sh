#!/bin/bash
# Copy build files for Singularity image building process

# Environment variables
source cicd/cicd.env
LOCAL_DIR="$(pwd)/"
SCIENCECLOUD="${SCIENCECLOUD_HOST}:${SCIENCECLOUD_DIR}"


# Copy necessary build files
scp "${LOCAL_DIR}/cicd/cicd.env" "${SCIENCECLOUD}"
scp "${LOCAL_DIR}/cicd/image.def" "${SCIENCECLOUD}"
scp "${LOCAL_DIR}/cicd/build_image.sh" "${SCIENCECLOUD}"
scp -r "${LOCAL_DIR}/requirements/" "${SCIENCECLOUD}"
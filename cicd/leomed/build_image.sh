#!/bin/bash
# Build script for Singularity image

# Environment variables
source cicd.env

# Build singularity image
sudo singularity build "${SCIENCECLOUD_DIR}/${IMAGE_NAME}" "${SCIENCECLOUD_DIR}/${IMAGE_FILE}"
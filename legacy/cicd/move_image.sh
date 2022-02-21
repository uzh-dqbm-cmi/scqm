#!/bin/bash

# Load environment variables
source cicd/cicd.env
echo "Please enter your LeoMed username:"
read -r LEOMED_USERNAME
LEOMED_HOST="${LEOMED_USERNAME}@${LEOMED_HOST}"

# Copy image
# Add -v flag for debugging in case of problems
scp -3 -v "${SCIENCECLOUD_HOST}:${SCIENCECLOUD_DIR}/${IMAGE_NAME}" "${LEOMED_HOST}:${LEOMED_DIR}/containers/${IMAGE_NAME}"
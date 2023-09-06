#!/bin/bash
set -euf -o pipefail
declare -r IMAGE_NAME="training_draft"
docker build -t $IMAGE_NAME .
docker run -u "$(id -u)":"$(id -g)" --gpus all --rm -it --mount type=bind,source="$(pwd)"/logs/,target=/usr/src/app/logs/ \
    --mount type=bind,source="$(pwd)"/data/,target=/usr/src/app/data/ \
    --mount type=bind,source="$(pwd)"/models/,target=/usr/src/app/models/ \
    $IMAGE_NAME python -m training_draft.train
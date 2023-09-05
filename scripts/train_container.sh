#!/bin/bash
set -euf -o pipefail
declare -r IMAGE_NAME="training_draft"
docker run -u "$(id -u)":"$(id -g)" --rm -it --mount type=bind,source="$(pwd)"/logs/,target=/usr/src/app/logs/ \
    --mount type=bind,source="$(pwd)"/data/,target=/usr/src/app/data/ \
    $IMAGE_NAME python -m training_draft.train
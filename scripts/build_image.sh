#!/bin/bash

# Get current user's ID and group ID
USER_ID=277762
GROUP_ID=30133
USER_NAME=aloureir

cd ..

# Build the Docker image
docker build \
  --build-arg USER_ID=$USER_ID \
  --build-arg GROUP_ID=$GROUP_ID \
  --build-arg USER_NAME=$USER_NAME \
  --build-arg WORKSPACE=/mnt/nfs/home/aloureir/moe-inference/ \
  --build-arg CACHE=/mnt/nfs/home/aloureir/cache/ \
  -t ic-registry.epfl.ch/sacs/aloureir/moe-gitlab:latest \
  -t moe-gitlab:latest .

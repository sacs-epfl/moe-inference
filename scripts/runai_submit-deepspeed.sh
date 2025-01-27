#! /bin/bash

    # --pvc runai-sacs-aloureir-scratch:/mnt/nfs \
runai submit \
    --name moe-topology-deepspeed \
    -i ic-registry.epfl.ch/sacs/aloureir/moe-deepspeed-gitlab:latest \
    --gpu 1 --cpu 1 \
    --pvc sacs-scratch:/mnt/nfs \
    --large-shm \
    --command \
    -- /mnt/nfs/home/aloureir/moe-inference/scripts/docker_startup.sh

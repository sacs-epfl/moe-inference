#! /bin/bash

    # --pvc runai-sacs-aloureir-scratch:/mnt/nfs \
    # --host-ipc \
runai submit \
    --name moe-topology \
    -i ic-registry.epfl.ch/sacs/aloureir/moe-gitlab:latest \
    --gpu 2 --cpu 1 \
    --pvc sacs-scratch:/mnt/nfs \
    --command \
    -- /mnt/nfs/home/aloureir/moe-inference/scripts/docker_startup.sh

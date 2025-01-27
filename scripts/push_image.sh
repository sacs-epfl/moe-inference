#! /usr/bin/bash

cd ..

docker login ic-registry.epfl.ch/sacs/aloureir/
docker push ic-registry.epfl.ch/sacs/aloureir/moe:latest

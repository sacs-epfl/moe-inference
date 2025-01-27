#! /bin/bash

PROFILE_NAME="../reports/profile-report-reduce-scatter-2"

  # --cuda-memory-usage=true \
  # --capture-range-end=repeat-shutdown:3 \
nsys profile \
  --trace=cuda,osrt,mpi,nvtx,cudnn,cublas \
  --output=${PROFILE_NAME} \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --cudabacktrace=true \
  --python-sampling=true \
  --osrt-threshold=10000 \
  --gpu-metrics-device \
  -x true \
  ./run_moe.sh
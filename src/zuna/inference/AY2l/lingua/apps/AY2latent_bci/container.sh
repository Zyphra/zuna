#!/bin/bash

docker run -t -d --name=AY2latent2_bci --network=host --ipc=host --privileged --shm-size=1800gb --gpus all --expose 2222 --rm  \
    # -v /nfsdata/hpcx:/opt/hpcx \ #for multi-node
    -v /mnt/shared/jonas/workspace/:/workspace \
    -v /mnt/shared/datasets:/workspace/datasets \
    # -v /raid0/cache:/cache \
    nvcr.io/nvidia/pytorch:25.03-py3 bash

tmux new -s AY2latent2_bci 'docker exec -it AY2latent2_bci bash -c "cd /workspace/ ; bash"'
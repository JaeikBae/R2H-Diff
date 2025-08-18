docker stop diffusion-hsi
docker build -t diffusion-hsi-img . && \
docker run \
    $1 \
    -it --rm \
    -v $(pwd):/app \
    --gpus all \
    --name diffusion-hsi \
    --shm-size=64G \
    diffusion-hsi-img \
    /bin/bash

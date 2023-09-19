## Build docker

```bash
sudo POLICY=manylinux_2_28 CUDA=11.6 PLATFORM=x86_64 TAG=latest bash ./build.sh
```

## Run docker

`docker run -it dgNN:v1 /bin/bash`

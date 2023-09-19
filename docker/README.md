This is a docker building pipeline for dgSPARSE-Lib with pip standard.

## Build docker

```bash
sudo POLICY=manylinux2014 CUDA=11.6 PLATFORM=x86_64 TAG=latest bash ./build.sh
```

## Run docker

`sudo docker run -it dgsparse/manylinux2014_x86_64 /bin/bash`

# Docker Usage Guide

`lmm` provides a heavily optimized Docker container.

## Running the Container

The default entrypoint is set to the CLI (`lmm`).

```sh
# Display help menu
docker run -it wiseaidev/lmm --help

# Perform a text continuation prediction
docker run -it wiseaidev/lmm predict --text "Wise AI built the first LMM" --window 10 --predict-length 180
```

## Pulling the Image

You can pull the official pre-built image from Docker Hub:

```sh
docker pull wiseaidev/lmm:latest
```

## Building the Container Locally

Install the [docker buildx plugin](https://docs.docker.com/build/concepts/overview/):

```sh
sudo apt-get update
sudo apt-get install docker-buildx-plugin
```

Once installed, you can use BuildKit natively using:

```sh
docker buildx build -t local/lmm .
docker run -it local/lmm --help
```

You can alias the command for convenience in your shell profile (e.g. `~/.bashrc` or `~/.zshrc`):

```sh
alias lmm="docker run -it wiseaidev/lmm"
```

This directory contains the container definitions for the project.

- `docker/` contains the Docker build definition used to build the main pipeline image.
- `apptainer/` contains Apptainer/Singularity definition files for sidecar or cluster-oriented images.
- The Docker build stages a minimal context under `deployment/containers/docker/.context`
  so `Dockerfile` and `.dockerignore` can live together in `docker/`.
- The main image installs from the root `pyproject.toml` and `uv.lock`.
- The Galician sidecar installs from `environments/gl-enrichment/pyproject.toml`
  and its independent lockfile so its newer Transformers stack stays isolated.

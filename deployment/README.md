Deployment-related assets live here.

- `containers/` contains image definition files.
- `build/` contains scripts that build Docker or Apptainer/Singularity images.
- `run/` contains portable tracked wrappers for running the pipeline locally or on shared infrastructure.

Local-only wrappers or cluster-specific helpers should stay outside `deployment/run/` so the tracked scripts remain portable.

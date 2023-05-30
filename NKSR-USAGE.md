# NKSR Package Documentation

## Pre-built wheels

We have the following wheels available:

|                | `cu117` | `cu118` |
|----------------|---------|---------|
| PyTorch 2.0.0  | ✅      | ✅      |
| PyTorch 1.13.0 | ✅      | ✅      |

## Data preparation

TODO: Draw two figures

## Examples

- `examples/recons_simple.py`

- `examples/recons_colored_mesh.py`

- `examples/recons_by_chunk.py`

> To prevent OOM, one last resort is to add `PYTORCH_NO_CUDA_MEMORY_CACHING=1` as environment variable!

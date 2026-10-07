# Installation

## Core learning and search

Python 3.10 is the verified baseline. Package metadata permit Python 3.11, but that interpreter and macOS/Linux runtime combinations have not been locally verified for this release. Begin in a fresh environment, not by upgrading a working research environment in place.

```console
conda env create -f environment.yml
conda activate citydeploy
```

Choose **one** PyTorch build:

```console
# CPU
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

```console
# NVIDIA GPU: Windows or Linux, with a compatible driver
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Then install the core:

```console
python -m pip install -e .
citydeploy doctor
citydeploy demo --all-samplers
```

The tested dependency snapshot is `constraints-tested.txt`. For that narrower set, append `-c constraints-tested.txt` to the project install command. This is not an OS-independent lockfile. The [official PyTorch previous-version guide](https://pytorch.org/get-started/previous-versions/) documents available 2.5.1 builds.

## Physical evaluation

```console
python -m pip install -e ".[radio]"
citydeploy doctor --radio --ray-device gpu
citydeploy radio-demo --ray-device gpu
```

The radio stack is deliberately pinned to Sionna RT 1.2.1, Mitsuba 3.7.1 and Dr.Jit 1.2.0. Do not independently upgrade one of these libraries. NVIDIA GPU ray tracing uses Mitsuba/Dr.Jit/OptiX, not TensorFlow GPU support. The radio extra does not require the full TensorFlow-based communications stack.

CPU radio tracing requires a working LLVM backend; core CPU model training does not. Apple MPS can be selected for neural models where supported, but it is not an OptiX radio device. Check the doctor output rather than assuming identical support across platforms.

A narrowly scoped Windows compatibility helper supports the verified Dr.Jit 1.2.0 binary on newer NVIDIA driver families. It validates the binary hash, modifies a separate cached copy of three PTX text templates, and does not overwrite installed packages. Unknown binaries fail closed. See [troubleshooting](troubleshooting.md).

## Scene construction

```console
python -m pip install -e ".[radio,scene]"
citydeploy build-scene --help
```

These optional dependencies are larger and have platform-dependent wheel availability. Geographic construction needs network access to public OSM services; terrain retrieval is optional. Training from cached feature tensors and evaluation of already packaged scenes do not need the scene-construction extra.

## Installation versus workspace

After installation, commands can run from another directory. Create it first, then run `citydeploy --workspace <directory> init`. Relative data/model/output paths in commands and plans resolve against that workspace. No path to the developer's machine is required. Editable configuration copies override bundled defaults.

For non-editable installation use `python -m pip install .`. Source and wheel installation use the same packaged defaults.

#!/bin/bash

# Ensure the script exits on the first error
set -e

# Install wheels in the correct order of dependencies
python3 -m pip install ./downloads/setuptools-70.2.0-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/attrs-23.2.0-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/certifi-2024.7.4-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/charset_normalizer-3.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/idna-3.7-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/typing_extensions-4.12.2-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/urllib3-2.2.2-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/click-8.1.7-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/filelock-3.15.4-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/jinja2-3.1.4-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/MarkupSafe-2.1.5-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/packaging-24.1-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/platformdirs-4.2.2-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/six-1.16.0-py2.py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/smmap-5.0.1-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/tqdm-4.66.4-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/multidict-6.0.5-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/yarl-1.9.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/attrs-23.2.0-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/gin_config-0.5.0-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/frozenlist-1.4.1-cp312-cp312-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/protobuf-5.27.2-cp38-abi3-manylinux2014_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/setproctitle-1.3.3-cp312-cp312-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/PyYAML-6.0.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/requests-2.32.3-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/networkx-3.3-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/numpy-2.0.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/psutil-6.0.0-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/sentry_sdk-2.8.0-py2.py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/fsspec-2024.6.1-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/nvidia_nccl_cu12-2.20.5-py3-none-manylinux2014_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/nvidia_nvjitlink_cu12-12.5.82-py3-none-manylinux2014_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/torch-2.3.1-cp312-cp312-manylinux1_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/torchvision-0.18.1-cp312-cp312-manylinux1_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/torchaudio-2.3.1-cp312-cp312-manylinux1_x86_64.whl  --force-reinstall --no-index --no-deps
python3 -m pip install ./downloads/torchmetrics-1.4.0.post0-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/torchfunc-0.2.0-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/lightning_utilities-0.11.3.post0-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/lightning-2.3.2-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/pytorch_lightning-2.3.2-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/mpmath-1.3.0-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/sympy-1.12.1-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/jinja2-3.1.4-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/docker_pycreds-0.4.0-py2.py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/gitdb-4.0.11-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/GitPython-3.1.43-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/wandb-0.17.4-py3-none-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/aiohttp-3.9.5-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --no-deps
python3 -m pip install ./downloads/aiosignal-1.3.1-py3-none-any.whl --no-index --no-deps
python3 -m pip install ./downloads/pillow-10.4.0-cp312-cp312-manylinux_2_28_x86_64.whl --no-index --no-deps


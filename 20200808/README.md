### Hardware
Each of the available Intel CPUs (Intel(R) Xeon(R) Platinum 8176 CPU @ 2.10GHz) had 28 threads each and the available GPU was Tesla V100 with 16 GiB of VRAM.

### Software
Tests were conducted in a custom GPU-enabled docker container with Ubuntu 20.04, python 3.8, and with CUDA 10.1, available for download here:

`docker pull mirekphd/ml-gpu-py38-cuda101-cust:20200808`

The library versions used for these tests:
- `lightgbm==3.0.0rc1`
- `xgboost==1.2.0rc1`

_Note: conclusions regarding lightgbm’s GPU performance may change after the algo switched over to the currently tested CUDA implementation (instead of the current OpenCL)._


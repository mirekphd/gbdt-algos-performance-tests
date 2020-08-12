### Hardware
Each of the available Intel CPUs (Intel(R) Xeon(R) Platinum 8176 CPU @ 2.10GHz) had 28 threads each and the available GPU was Tesla V100 with 16 GiB of VRAM.

### Software
Tests were conducted in three different custom GPU-enabled docker containers:

- `docker pull mirekphd/ml-gpu-py38-cuda101-cust:20200809`,
containing these library versions used in the tests:
- `lightgbm==3.0.0rc1`
- `xgboost==1.2.0rc1`
- CUDA: 10.1

- `docker pull mirekphd/ml-gpu-py38-cuda101-cust:20200807`,
containing these library versions used in the tests:
- `lightgbm==2.3.1`
- `xgboost==1.1.1`
- CUDA: 10.1

- `docker pull mirekphd/ml-gpu-py36-cuda90:20191013`,
containing these library versions used in the tests:
- `lightgbm==2.2.3`
- `xgboost==0.81`
- CUDA: 9.0

_Note: conclusions regarding lightgbm’s GPU performance may change after the algo switched over to the currently tested CUDA implementation (instead of the current OpenCL)._


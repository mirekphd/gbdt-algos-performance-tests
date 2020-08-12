## Performance comparison of competing GBDT algorithms

I compared computational efficiency of the latest versions of GBDT algos - lightgbm and xgboost – in GPU "alone" (i.e. with a single CPU thread) and in CPU alone (using from 1 to 64 threads).

### Conclusions

- lightgbm has an excellent CPU implementation, while xgboost - an excellent GPU one, but one should rather avoid using lightgbm in GPU or xgboost in CPU alone,

- in CPU alone lightgbm is substantially faster than xgboost, in fact comparable to (or only slightly slower than) xgboost in the GPU,

- lightgbm in the CPU alone with 16 threads limit is the optimal configuration for the most frequently encountered tabular data sets,

- lightgbm above 16 CPU threads stops to scale well (so it is better to start a second model training process with the same optimal number of threads),

- using GPU in case of lightgbm currently does not improve performance if one has access to a CPU with at least 8 threads, which would on its own provide better performance regardless of data shape (note that combining GPU with multi-threaded CPU computations will obscure this issue),

- version 3.0 (RC1) of lightgbm is faster than either version 2.3 or 2.2, especially for very "tall" data shapes (with as least 100k times more rows than columns), where there is a 2x speed improvement over previous versions (when using at least 8 CPU threads),

- using GPU in case of xgboost is essential, because it is substantially slower in the CPU than lightgbm and scales poorly with more threads (in fact the value added from increasing the number of threads above 16 is negative),

- the optimal number of CPU threads for xgboost is also 8-16, but one should avoid using this algo in the CPU (in GPU it is significantly faster - one of the largest performance benefits observed in this study),

- GPU version of xgboost is the fastest option for this algo (except for very small data sets, because small data transfers to the GPU take too long in relation to GPU computations), but one should bear in mind rather restrictive memory limits (e.g. 16 GiB of VRAM in case of V100 or 11 GiB for 1080Ti), which restrict GPU use,

- the 16 GiB of video memory available in Tesla V100 GPUs allows for stable model training in data sets not exceeding 100m cells (e.g. 1m rows by 100 cols),

### Methodology

Results were averaged over multiple models (in total over 180 thousand) trained on simulated (random) binary classification data sets.

The data sets had different shapes, because computation times vary with data set shape, ranging from 10k to 10m rows and from 10 to 10k columns, excluding combinations that exceeded 100m cells (because they have proven too time- and/or memory to allow for stable model training in case of xgboost (OOM conditions for VRAM led in xgboost to frequent crashes of the entire python kernel).

To ensure comparability across algos all models – binary classifiers - had default hyperparmeters (including a fast learning rate of 0.1) except for:
- max_bin=63 (reduced to improve GPU training speeds, as per lightgbm recommendations - see [GPU Tuning Guide](https://lightgbm.readthedocs.io/en/latest/GPU-Performance.html)), 
- 1000 boosting rounds, without early stopping (increased to ensure meaningful training times, reducing impact of warm-up periods).

Comparability of model forecasting acurracy (across different data sets and algos) was guaranteed by the use of randomly simulated data, because as a result of this randomness all models had roughly the same accuracy (AUC of around 50%).

I did not compare the more realistic scenario of combined use of multi-threading in the CPU and GPU, to be able to measure the GPU performance independently from CPU.

The comparisons of model training times are only justified within the same data shape (not just the total number of cells), because training times differ significantly between different shapes (cf. the negative impact of increasing the number of columns).

### Hardware and software
The hardware and software used for these tests are specified independently within each versioned subfolder with test results.

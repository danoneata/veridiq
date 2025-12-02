To improve calibration, train an ensemble of networks:

```bash
for s in 1 2 3; do
    python veridiq/linear_probing/train_test.py --config_path veridiq/linear_probing/configs/av1m-clip-conv-seed-${s}.yaml
done
```

Predict on multiple test datasets:

```bash
for s in 1 2 3; do
    for d in av1m favc avlips bitdf; do
        python veridiq/calibration/predict.py -c av1m-clip-conv-seed-${s} -d $d -f clip
    done
done
```

Compile results (for a single model):

```bash
python veridiq/calibration/get_results.py -c av1m-clip-conv-seed-1
```

Compile results (for the ensemble):

```bash
python veridiq/calibration/get_results.py \
    -c av1m-clip-conv-seed-1 \
    -c av1m-clip-conv-seed-2 \
    -c av1m-clip-conv-seed-3
```
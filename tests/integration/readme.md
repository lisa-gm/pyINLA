# How to run the integration tests

## On Daint
I recomend getting on an interactive session where you can directly run the integration tests for all configurtions (in particular type sof parallelization).
```bash
srun --pty --partition=debug --account=xxxx bash
```

Then, you can run the integration tests sequentially using:
```bash
python runner.py
```

Further configurations are available in the `runner.py` file.


## On Fritz
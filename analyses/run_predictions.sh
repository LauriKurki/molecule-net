#!/bin/bash

python -m analyses.make_predictions \
    --workdir /Users/kurkil1/work/molnet/runs/attention-adam-3e-4 \
    --outputdir /Users/kurkil1/work/molnet/runs/attention-adam-3e-4/analysis_0.5 \
    --num_batches 4 \
    --batch_size 24 \
    --peak_threshold 0.5 \
    --old True 


#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

python -m training.sae.concurrent --config=topk.shakespeare_64x4 --load_from=shakespeare_64x4
#python -m training.sae.concurrent --config=topk_fat.shakespeare_64x4 --load_from=shakespeare_64x4
#python -m training.sae.concurrent --config=topk_staircase.shakespeare_64x4 --load_from=shakespeare_64x4

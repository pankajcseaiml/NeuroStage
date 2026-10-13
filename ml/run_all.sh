#!/usr/bin/env bash
# Full ML pipeline on synthetic data (use --data sleepedf for the real dataset)
set -e
DATA=${1:-synthetic}
python train.py --role baseline --data $DATA
python train.py --role teacher  --data $DATA
python train.py --role student  --data $DATA --teacher artifacts/teacher.pt   # knowledge distillation
python compress.py
python benchmark.py

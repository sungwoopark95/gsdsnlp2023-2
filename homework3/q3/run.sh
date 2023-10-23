#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python3 run.py train --train-src=./data/train.je --train-tgt=./data/train.ko --dev-src=./data/dev.je --dev-tgt=./data/dev.ko --vocab=vocab.json --cuda --lr=5e-4 --patience=1 --valid-niter=200 --batch-size=32 --dropout=.3
elif [ "$1" = "test" ]; then
	if [ -z "$2" ]; then 
		beam_size=10
	else
		beam_size=$2
	fi
    CUDA_VISIBLE_DEVICES=0 python3 run.py decode --beam-size $beam_size model.bin ./data/test.je ./data/test.ko outputs/test_outputs_${beam_size}.txt --cuda
elif [ "$1" = "train_local" ]; then
	python3 run.py train --train-src=./data/train.je --train-tgt=./data/train.ko --dev-src=./data/dev.je --dev-tgt=./data/dev.ko --vocab=vocab.json --lr=5e-4
elif [ "$1" = "test_local" ]; then
    python3 run.py decode model.bin ./data/test.je ./data/test.ko outputs/test_outputs.txt
elif [ "$1" = "vocab" ]; then
	python3 vocab.py --train-src=./data/train.je --train-tgt=./data/train.ko vocab.json		
else
	echo "Invalid Option Selected"
fi

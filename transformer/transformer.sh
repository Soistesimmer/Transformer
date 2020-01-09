#!/usr/bin/env bash
BASE_DIR=/home/wangante/google/official/transformer/
GPU_ID=0
# Ensure that PYTHONPATH is correctly defined as described in
# https://github.com/tensorflow/models/tree/master/official#requirements
# export PYTHONPATH="$PYTHONPATH:/path/to/models"

# Export variables
PARAM_SET=base
DATA_DIR=$BASE_DIR/data
MODEL_DIR=$BASE_DIR/model/model_$PARAM_SET
VOCAB_FILE=$DATA_DIR/vocab.ende.32768

# Download training/evaluation/test datasets
python data_download.py --data_dir=$DATA_DIR

# Train the model for 10 epochs, and evaluate after every epoch.
python3 v2/transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
    --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET \
    --bleu_source=$DATA_DIR/newstest2014.en --bleu_ref=$DATA_DIR/newstest2014.de #> transformer.log 2>&1 &

## Run during training in a separate process to get continuous updates,
## or after training is complete.
#tensorboard --logdir=$MODEL_DIR
#
## Translate some text using the trained model
#python v2/translate.py --model_dir=$MODEL_DIR --vocab_file=$VOCAB_FILE \
#    --param_set=$PARAM_SET --text="hello world"
#
## Compute model's BLEU score using the newstest2014 dataset.
#python v2/translate.py --model_dir=$MODEL_DIR --vocab_file=$VOCAB_FILE \
#    --param_set=$PARAM_SET --file=$DATA_DIR/newstest2014.en --file_out=translation.en
#python compute_bleu.py --translation=translation.en --reference=$DATA_DIR/newstest2014.de

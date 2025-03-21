DATASET=marco
DATA_PATH=
OUTPUT_DIR=
RESULTS_FILE=
CKPT_PATH=
python3 ./GR_train/test.py \
    --gpu_id 0 \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 64 \
    --num_beams 100 \
    --test_prompt_ids 0 \
    --index_file .index.json
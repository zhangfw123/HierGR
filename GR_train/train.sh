
DATASET=marco
OUTPUT_DIR=
torchrun --nproc_per_node=4 --master_port=2314 ./GR_train/finetune.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --per_device_batch_size 256 \
    --learning_rate 5e-4 \
    --epochs 80 \
    --index_file .index.json \
    --temperature 1.0 \
    --dataset $DATASET \
    --data_path  data \
    --base_model google-t5/t5-base
    
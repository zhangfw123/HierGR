python ./RQ-VAE/main.py \
  --device cuda \
  --data_path embedding_data_path \
  --alpha 0.01 \
  --beta 0.0001 \
  --ckpt_dir ckpt_dir \
  --eval_step 10\
  --epochs 300\
  --batch_size 2048\
  --num_emb_list 256 256 256 256
  
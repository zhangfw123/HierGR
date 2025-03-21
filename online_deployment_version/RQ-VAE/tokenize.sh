data_split=$1
python ./generate_indices.py \
  --output_dir output_dir \
  --checkpoint ckpt \
  --data_split $data_split
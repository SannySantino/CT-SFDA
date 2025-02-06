python -u run.py \
  --task_name classification \
  --is_training 0 \
  --root_path ./HAR/ \
  --model_id 7to13 \
  --model SFDA \
  --data HAR \
  --batch_size 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.005 \
  --input_channels 9\
  --num_classes 6\
  --train_domain train_13.pt\
  --test_domain test_13.pt\
  --stage 'tta'\
  --train_epochs 8 \
  --win_size 128\
  --win_step 128\
  --tta 1\
  --delta 0.002\
  --N 4

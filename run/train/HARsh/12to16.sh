python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./HAR/ \
  --model_id 12to16 \
  --model SFDA \
  --data HAR \
  --batch_size 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --input_channels 9\
  --kernel_size 3\
  --stride 1\
  --num_classes 6\
  --train_domain train_12.pt\
  --test_domain test_12.pt\
  --stage 'stage1'\
  --train_epochs 15 \
  --patience 20 \
  --win_size 128\
  --win_step 128\




python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./HAR/ \
  --model_id 12to16 \
  --model SFDA \
  --data HAR \
  --batch_size 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.005 \
  --input_channels 9\
  --kernel_size 3\
  --stride 1\
  --num_classes 6\
  --train_domain train_12.pt\
  --test_domain test_12.pt\
  --stage 'stage2'\
  --train_epochs 15 \
  --patience 20 \
  --win_size 128\
  --win_step 128\


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./HAR/ \
  --model_id 12to16 \
  --model SFDA \
  --data HAR \
  --batch_size 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.005 \
  --input_channels 9\
  --kernel_size 3\
  --stride 1\
  --num_classes 6\
  --train_domain train_16.pt\
  --test_domain train_16.pt\
  --stage 'stage3'\
  --train_epochs 8 \
  --patience 20 \
  --win_size 128\
  --win_step 128\

python -u run.py \
  --task_name classification \
  --is_training 0 \
  --root_path ./HAR/ \
  --model_id 12to16 \
  --model SFDA \
  --data HAR \
  --batch_size 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.005 \
  --input_channels 9\
  --kernel_size 3\
  --stride 1\
  --num_classes 6\
  --train_domain train_16.pt\
  --test_domain test_16.pt\
  --stage 'stage3'\
  --train_epochs 8 \
  --patience 20 \
  --win_size 128\
  --win_step 128\

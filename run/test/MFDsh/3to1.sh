# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./FD/ \
#   --model_id 3to1 \
#   --model SFDA \
#   --data FDA \
#   --batch_size 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.005 \
#   --input_channels 10\
#   --kernel_size 3\
#   --stride 1\
#   --num_classes 3\
#   --win_size 5120\
#   --win_step 5120\
#   --train_domain train_3.pt\
#   --test_domain test_3.pt\
#   --stage 'stage1'\
#   --train_epochs 8



# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./FD/ \
#   --model_id 3to1 \
#   --model SFDA \
#   --data FDA \
#   --batch_size 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.002 \
#   --input_channels 10\
#   --kernel_size 3\
#   --stride 1\
#   --num_classes 3\
#   --win_size 5120\
#   --win_step 5120\
#   --train_domain train_3.pt\
#   --test_domain test_3.pt\
#   --stage 'stage2'\
#   --train_epochs 20


# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./FD/ \
#   --model_id 3to1 \
#   --model SFDA \
#   --data FDA \
#   --batch_size 32 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.005 \
#   --input_channels 10\
#   --kernel_size 3\
#   --stride 1\
#   --num_classes 3\
#   --win_size 5120\
#   --win_step 5120\
#   --train_domain train_1.pt\
#   --test_domain train_1.pt\
#   --stage 'stage3'\
#   --train_epochs 8


  python -u run.py \
  --task_name classification \
  --is_training 0 \
  --root_path ./FD/ \
  --model_id 3to1 \
  --model SFDA \
  --data FDA \
  --batch_size 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.005 \
  --input_channels 10\
  --kernel_size 3\
  --stride 1\
  --num_classes 3\
  --win_size 5120\
  --win_step 5120\
  --train_domain train_1.pt\
  --test_domain test_1.pt\
  --stage 'stage3'\
  --train_epochs 8


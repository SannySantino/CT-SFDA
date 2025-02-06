python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./SSC/ \
  --model_id 0to11 \
  --model SFDA \
  --data SSC \
  --batch_size 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.005 \
  --kernel_size 3\
  --stride 1\
  --num_classes 5\
  --train_domain train_0.pt\
  --test_domain test_0.pt\
  --stage 'stage1'\
  --train_epochs 8 \
  --win_size 3072\
  --win_step 3072


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./SSC/ \
  --model_id 0to11 \
  --model SFDA \
  --data SSC \
  --batch_size 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.002 \
  --kernel_size 3\
  --stride 1\
  --num_classes 5\
  --train_domain train_0.pt\
  --test_domain test_0.pt\
  --stage 'stage2'\
  --train_epochs 15 \
  --win_size 3072\
  --win_step 3072


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./SSC/ \
  --model_id 0to11 \
  --model SFDA \
  --data SSC \
  --batch_size 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.005 \
  --kernel_size 3\
  --stride 1\
  --num_classes 5\
  --train_domain train_11.pt\
  --test_domain train_11.pt\
  --stage 'stage3'\
  --train_epochs 8 \
  --win_size 3072\
  --win_step 3072

python -u run.py \
  --task_name classification \
  --is_training 0 \
  --root_path ./SSC/ \
  --model_id 0to11 \
  --model SFDA \
  --data SSC \
  --batch_size 32 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.005 \
  --kernel_size 3\
  --stride 1\
  --num_classes 5\
  --train_domain train_11.pt\
  --test_domain test_11.pt\
  --stage 'stage3'\
  --train_epochs 8 \
  --win_size 3072\
  --win_step 3072



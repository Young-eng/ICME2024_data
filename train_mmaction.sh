#! /bin/bash


#SBATCH --job-name=NoBug


#SBATCH --output y.out

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH -p 3090 --gres=gpu:3090:1 

#SBATCH --time=30:00:00



CUDA_VISIBLE_DEVICES=0 

OUTPUT_DIR='path_to/VideoMAE/output'
# path to Kinetics set (train.csv/val.csv/test.csv)
DATA_PATH='path_to/action_recognition/annotation/'
# path to pretrain model
MODEL_PATH='path_to/VideoMAE/ckpts/checkpoint.pth'

OMP_NUM_THREADS=1

python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port 12320 --nnodes=1  --node_rank=0 \
    run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 140 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 4 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 100 \
    --dist_eval \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --num_workers 4    





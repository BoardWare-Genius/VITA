#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`


export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
# export NCCL_IB_GID_INDEX=3
#export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
#export NCCL_IB_SL=3
#export NCCL_CHECKS_DISABLE=1
export NCCL_P2P_DISABLE=0
#export NCCL_LL_THRESHOLD=16384
# export NCCL_IB_CUDA_SUPPORT=1
export NCCL_DEBUG=INFO

INDEX=0
MASTER_ADDR="127.0.0.1"
# communication on taiji platform
DISTRIBUTED_ARGS="
    --nproc_per_node 2 \
    --nnodes 1 \
    --node_rank $INDEX \
    --master_addr $MASTER_ADDR \
    --master_port 9999
"
export NCCL_TIMEOUT=25200
MODEL_TYPE=qwen2p5_instruct
BASE_DIR=/Model/Weight/VITA-MLLM/VITA-1.5
OUTPUT_DIR=$1
OUTPUT_DIR_FT=${OUTPUT_DIR}/llava-s3-finetune_task_neg
mkdir -p ${OUTPUT_DIR_FT}

torchrun $DISTRIBUTED_ARGS vita/train/train.py \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path ${BASE_DIR}\
    --model_type $MODEL_TYPE \
    --version qwen2p5_instruct \
    --dataset_use Pretrain_video \
    --vision_tower ${BASE_DIR}/InternViT-300M-448px \
    --mm_projector_type mlp2x_gelu \
    --audio_encoder ${BASE_DIR}/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning \
    --freeze_audio_encoder True \
    --freeze_audio_encoder_adapter True \
    --freeze_backbone False \
    --tune_mm_mlp_adapter False \
    --tune_audio_mlp_adapter False \
    --image_aspect_ratio square \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR_FT} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to none \
    2>&1 | tee -a ${OUTPUT_DIR_FT}/log_node_$INDEX.txt && echo "Done."






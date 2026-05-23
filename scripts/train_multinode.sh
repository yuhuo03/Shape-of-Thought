#!/bin/bash
export PYTHONPATH=YOUR_PYTHONPATH
export WANDB_API_KEY=YOUR_WANDB_API_KEY

NUM_NODES=YOUR_NUM_NODES
NODE_RANK=YOUR_NODE_RANK
NPROC_PER_NODE=YOUR_NPROC_PER_NODE
MASTER_ADDR=YOUR_MASTER_ADDR
MASTER_PORT=YOUR_MASTER_PORT
MODEL_PATH=YOUR_MODEL_PATH

echo "[DDP] NNODES=$NUM_NODES NODE_RANK=$NODE_RANK NPROC_PER_NODE=$NPROC_PER_NODE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

OUT="run_node${NODE_RANK}.out"
ERR="run_node${NODE_RANK}.err"

# replace the variables with your own
torchrun \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  --nproc_per_node=$NPROC_PER_NODE \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --model_path $MODEL_PATH \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from $MODEL_PATH \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 1 \
  --lr 2e-5 \
  --lr_scheduler cosine \
  --min_lr 1e-6 \
  --num_worker 1 \
  --expected_num_tokens 40000 \
  --max_num_tokens 50000 \
  --max_num_tokens_per_sample 50000 \
  --prefer_buffer_before 20000 \
  --num_shard=$(($NUM_NODES * $NPROC_PER_NODE)) \
  --sharding_strategy="HYBRID_SHARD" \
  --wandb_project "Shape-of-Thought" \
  --wandb_name "h100-sot-n${NUM_NODES}-r${NODE_RANK}-$(date +%Y%m%d_%H%M%S)" \
  --cpu_offload True \
  --save_every 300 \
  --warmup_steps 50 \
  --total_steps 10000 \
  --results_dir results/ \
  --checkpoint_dir results/checkpoints/ \
  > "$OUT" 2> "$ERR"

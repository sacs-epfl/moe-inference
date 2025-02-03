#! /usr/bin/bash
cd ../src

python3 start.py \
    --model_type "encoder-decoder" \
    --batch_size 250 \
    --seq_len 120 \
    --num_experts 128 \
    --num_parallel_experts_per_GPU 1 \
    --num_iters 100 \
    --profile False \
    --nvidia_profile False \
    --world 1 \
    --dataset "bookcorpus" \
    --enable_router_skew False \
    --router_skew 0.6 \
    --num_experts_skew 2 \
    --expert_manager "MegaBlocks" \
    --path "../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_1_gpus_normalrun_get_tokens_per_exp" \
    --label "Encoder-Decoder MegaBlocks 128 Experts" \
    --experiment_name "encoder-decoder_mgblcks_128_experts_4_gpus_normalrun"

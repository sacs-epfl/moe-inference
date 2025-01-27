dataset=bookcorpus
num_iters=100

num_gpus=4

cd ../src

deepspeed --num_gpus $num_gpus start_deepspeed.py \
        --dataset $dataset \
        --num_iters $num_iters \
        --batch_size 250 \
        --seq_len 120 \
        --num_experts 128 \
        --model_type "encoder-decoder" \
        --router_skew 0.6 \
        --num_experts_skew 20 \
        --enable_router_skew False \
        --capacity_factor 50.0 \
        --world_size $num_gpus \
        -en "encoder-decoder_deepspeed_128_experts_4_gpus_normalrun" \
        --path "../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_normalrun"



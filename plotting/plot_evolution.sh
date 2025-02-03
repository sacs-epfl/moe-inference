#!/usr/bin/zsh

# python plot_experiment.py \
#     --type "evolution" \
#     --evolution_type "batch_size" \
#     --evolution_label "Batch Size" \
#     --rest "../outputs/encoder_deepspeed_128_exps_6_GPUS_reduce_scatter_10_bs    ../outputs/encoder_mgblcks_128_exps_6_GPUS_reduce_scatter_10_bs    ../outputs/encoder_sliced_1st_optim_128_exps_6_GPUS_reduce_scatter_10_bs    \
#           ../outputs/encoder_deepspeed_128_exps_6_GPUS_reduce_scatter_100_bs    ../outputs/encoder_mgblcks_128_exps_6_GPUS_reduce_scatter_100_bs    ../outputs/encoder_sliced_1st_optim_128_exps_6_GPUS_reduce_scatter_100_bs    \
#           ../outputs/encoder_deepspeed_128_exps_6_GPUS_new_code    ../outputs/encoder_mgblcks_128_exps_6_GPUS_reduce_scatter    ../outputs/encoder_sliced_1st_optim_128_exps_6_GPUS_reduce_scatter    \
#           ../outputs/encoder_deepspeed_128_exps_6_GPUS_reduce_scatter_500_bs    ../outputs/encoder_mgblcks_128_exps_6_GPUS_reduce_scatter_500_bs    ../outputs/encoder_sliced_1st_optim_128_exps_6_GPUS_reduce_scatter_500_bs \
#     ../outputs/encoder-decoder_deepspeed_128_exps_6_GPUS_reduce_scatter_10_bs    ../outputs/encoder-decoder_mgblcks_128_exps_6_GPUS_reduce_scatter_10_bs    ../outputs/encoder-decoder_sliced_1st_optim_128_exps_6_GPUS_reduce_scatter_10_bs    \
#           ../outputs/encoder-decoder_deepspeed_128_exps_6_GPUS_reduce_scatter_100_bs    ../outputs/encoder-decoder_mgblcks_128_exps_6_GPUS_reduce_scatter_100_bs    ../outputs/encoder-decoder_sliced_1st_optim_128_exps_6_GPUS_reduce_scatter_100_bs    \
#           ../outputs/encoder-decoder_deepspeed_128_exps_6_GPUS_new_code    ../outputs/encoder-decoder_mgblcks_128_exps_6_GPUS_reduce_scatter    ../outputs/encoder-decoder_sliced_1st_optim_128_exps_6_GPUS_reduce_scatter    \
#           ../outputs/encoder-decoder_deepspeed_128_exps_6_GPUS_reduce_scatter_500_bs    ../outputs/encoder-decoder_mgblcks_128_exps_6_GPUS_reduce_scatter_500_bs    ../outputs/encoder-decoder_sliced_1st_optim_128_exps_6_GPUS_reduce_scatter_500_bs   "

# python plot_experiment.py \
#     --type "evolution" \
#     --evolution_type "num_experts" \
#     --evolution_label "Number of Experts" \
#     --rest "../outputs/mgblcks/encoder_mgblcks_8_experts_4_gpus    ../outputs/mgblcks/encoder_mgblcks_16_experts_4_gpus    ../outputs/mgblcks/encoder_mgblcks_32_experts_4_gpus    ../outputs/mgblcks/encoder_mgblcks_64_experts_4_gpus    ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus    ../outputs/mgblcks/encoder_mgblcks_256_experts_4_gpus \
#     ../outputs/deepspeed/encoder-decoder_deepspeed_8_experts_4_gpus    ../outputs/deepspeed/encoder-decoder_deepspeed_16_experts_4_gpus    ../outputs/deepspeed/encoder-decoder_deepspeed_32_experts_4_gpus    ../outputs/deepspeed/encoder-decoder_deepspeed_64_experts_4_gpus    ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus    ../outputs/deepspeed/encoder-decoder_deepspeed_256_experts_4_gpus    \
#     ../outputs/mgblcks/encoder-decoder_mgblcks_8_experts_4_gpus   ../outputs/mgblcks/encoder-decoder_mgblcks_16_experts_4_gpus    ../outputs/mgblcks/encoder-decoder_mgblcks_32_experts_4_gpus    ../outputs/mgblcks/encoder-decoder_mgblcks_64_experts_4_gpus    ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus    ../outputs/mgblcks/encoder-decoder_mgblcks_256_experts_4_gpus \
#     ../outputs/sliced/encoder_sliced_8_experts_4_gpus    ../outputs/sliced/encoder_sliced_16_experts_4_gpus    ../outputs/sliced/encoder_sliced_32_experts_4_gpus    ../outputs/sliced/encoder_sliced_64_experts_4_gpus    ../outputs/sliced/encoder_sliced_128_experts_4_gpus    ../outputs/sliced/encoder_sliced_256_experts_4_gpus    \
#     ../outputs/sliced/encoder-decoder_sliced_8_experts_4_gpus    ../outputs/sliced/encoder-decoder_sliced_16_experts_4_gpus    ../outputs/sliced/encoder-decoder_sliced_32_experts_4_gpus    ../outputs/sliced/encoder-decoder_sliced_64_experts_4_gpus    ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus    ../outputs/sliced/encoder-decoder_sliced_256_experts_4_gpus \
#     ../outputs/deepspeed/encoder_deepspeed_8_experts_4_gpus   ../outputs/deepspeed/encoder_deepspeed_16_experts_4_gpus    ../outputs/deepspeed/encoder_deepspeed_32_experts_4_gpus    ../outputs/deepspeed/encoder_deepspeed_64_experts_4_gpus    ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus    ../outputs/deepspeed/encoder_deepspeed_256_experts_4_gpus    "

# python plot_experiment.py \
#     --type "evolution" \
#     --evolution_type "batch_size" \
#     --evolution_label "Batch Size" \
#     --rest " ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_10_bs    ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_100_bs    ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus    ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_450_bs \
#     ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_10_bs    ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_100_bs    ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus    ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_450_bs   \
#     ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_10_bs    ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_100_bs    ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus    ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_450_bs \
#     ../outputs/sliced/encoder_sliced_128_experts_4_gpus_10_bs    ../outputs/sliced/encoder_sliced_128_experts_4_gpus_100_bs    ../outputs/sliced/encoder_sliced_128_experts_4_gpus    ../outputs/sliced/encoder_sliced_128_experts_4_gpus_450_bs \
#     ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_10_bs    ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_100_bs    ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus    ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_450_bs \
#    ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_10_bs    ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_100_bs    ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus    ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_450_bs"

# python plot_experiment.py \
#     --type "evolution" \
#     --evolution_type "num_experts_skew" \
#     --evolution_label "Number of Skewed Experts" \
#     --rest " ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_0.01_nes    ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_0.2_nes  ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_0.5_nes ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_0.6_nes ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus    ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_0.8_nes \
#     ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_0.01_nes    ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_0.2_nes  ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_0.5_nes  ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_0.6_nes ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_0.6_rs_90_cf   ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_0.8_nes   \
#     ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_0.01_nes    ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_0.2_nes  ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_0.5_nes ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_0.6_nes  ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus    ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_0.8_nes \
#     ../outputs/sliced/encoder_sliced_128_experts_4_gpus_0.01_nes    ../outputs/sliced/encoder_sliced_128_experts_4_gpus_0.2_nes  ../outputs/sliced/encoder_sliced_128_experts_4_gpus_0.5_nes ../outputs/sliced/encoder_sliced_128_experts_4_gpus_0.6_nes  ../outputs/sliced/encoder_sliced_128_experts_4_gpus    ../outputs/sliced/encoder_sliced_128_experts_4_gpus_0.8_nes \
#     ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_0.01_nes    ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_0.2_nes  ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_0.5_nes  ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_0.6_nes ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus    ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_0.8_nes \
#    ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_0.01_nes    ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_0.2_nes  ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_0.5_nes  ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_0.6_nes ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_0.6_rs_90_cf   ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_0.8_nes"

python plot_experiment.py \
    --type "evolution" \
    --evolution_type "router_skew" \
    --evolution_label "Router Skew" \
    --rest " ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_0.0_rs    ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_0.2_rs  ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_0.4_rs  ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus    ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_0.8_rs \
    ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_0.0_rs    ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_0.2_rs  ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_0.4_rs  ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus   ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_0.8_rs   \
    ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_0.0_rs    ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_0.2_rs  ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_0.4_rs  ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus    ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_0.8_rs \
    ../outputs/sliced/encoder_sliced_128_experts_4_gpus_0.0_rs    ../outputs/sliced/encoder_sliced_128_experts_4_gpus_0.2_rs  ../outputs/sliced/encoder_sliced_128_experts_4_gpus_0.4_rs  ../outputs/sliced/encoder_sliced_128_experts_4_gpus ../outputs/sliced/encoder_sliced_128_experts_4_gpus_0.8_rs \
    ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_0.0_rs    ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_0.2_rs  ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_0.4_rs  ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus    ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_0.8_rs \
   ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_0.0_rs_old    ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_0.2_rs_old  ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_0.4_rs_old  ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus   ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_0.8_rs_old"
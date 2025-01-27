#!/usr/bin/zsh

# python plot_experiment.py \
#     --type "evolution" \
#     --evolution_values 8 16 32 64 128 \
#     --evolution_objects  "Encoder Sliced with optimization" "Deepspeed" \
#     --comparison_object "Deepspeed" \
#     --metric "Number of experts" \
#     --comparisons "encoder_only_simple_deepspeed" "encoder_only_simple_deepspeed_16experts" "encoder_deepspeed_32_experts_0_parallel" "encoder_deepspeed_64_experts_0_parallel" "encoder_deepspeed_128_experts_0_parallel" \
#     --caching "save" \
#     --dirs "[['../outputs/encoder_sliced_1st_optim_8_experts_0_parallel', '../outputs/encoder_simple_deepspeed_8_experts'], \
#         ['../outputs/encoder_sliced_1st_optim_16_experts_0_parallel', '../outputs/encoder_simple_deepspeed_16_experts'], \
#         ['../outputs/encoder_sliced_1st_optim_32_experts_0_parallel', '../outputs/encoder_deepspeed_32_experts_0_parallel'], \
#         ['../outputs/encoder_sliced_1st_optim_64_experts_0_parallel', '../outputs/encoder_deepspeed_64_experts_0_parallel'], \
#         ['../outputs/encoder_sliced_1st_optim_128_experts_0_parallel', '../outputs/encoder_deepspeed_128_experts_0_parallel']]" 

# python plot_experiment.py \
#     --type "evolution" \
#     --evolution_values 8 16 32 64 128 \
#     --evolution_objects  "Encoder-Decoder Sliced with optimization" "Deepspeed" \
#     --comparison_object "Deepspeed" \
#     --metric "Number of experts" \
#     --comparisons "encoder_decoder_simple_deepspeed" "encoder_decoder_simple_deepspeed_16experts" "encoder-decoder_deepspeed_32_experts_0_parallel" "encoder-decoder_deepspeed_64_experts_0_parallel" "encoder-decoder_deepspeed_128_experts_0_parallel" \
#     --caching "load" "final" \
#     --dirs "[['../outputs/encoder-decoder_sliced_1st_optim_8_experts_0_parallel', '../outputs/encoder-decoder_simple_deepspeed_8_experts'], \
#         ['../outputs/encoder-decoder_sliced_1st_optim_16_experts_0_parallel', '../outputs/encoder-decoder_simple_deepspeed_16_experts'], \
#         ['../outputs/encoder-decoder_sliced_1st_optim_32_experts_0_parallel', '../outputs/encoder-decoder_deepspeed_32_experts_0_parallel'], \
#         ['../outputs/encoder-decoder_sliced_1st_optim_64_experts_0_parallel', '../outputs/encoder-decoder_deepspeed_64_experts_0_parallel'], \
#         ['../outputs/encoder-decoder_sliced_1st_optim_128_experts_0_parallel', '../outputs/encoder-decoder_deepspeed_128_experts_0_parallel']]" 

# python plot_experiment.py \
#     --type "evolution" \
#     --evolution_values 8 16 32 64 128 256\
#     --evolution_objects  "Encoder Sliced with optimization" "Deepspeed" "Encoder Megablocks" "Encoder Megablocks optimized"\
#     --comparison_object "Deepspeed" \
#     --metric "Number of Experts" \
#     --comparisons "encoder_deepspeed_8_experts_6_GPUS" "encoder_deepspeed_16_experts_6_GPUS" "encoder_deepspeed_32_experts_6_GPUS" "encoder_deepspeed_64_experts_6_GPUS" "encoder_deepspeed_128_experts_6_GPUS" "encoder_deepspeed_256_experts_6_GPUS" \
#     --caching "save" \
#     --dirs "[['../outputs/encoder_sliced_1st_optim_8_experts_6_GPUS', '../outputs/encoder_deepspeed_8_experts_6_GPUS', '../outputs/encoder_megablocks_8_experts_6_GPUS', '../outputs/encoder_megablocks_8_experts_6_GPUS_optimized_loops'], \
#         ['../outputs/encoder_sliced_1st_optim_16_experts_6_GPUS', '../outputs/encoder_deepspeed_16_experts_6_GPUS', '../outputs/encoder_megablocks_16_experts_6_GPUS', '../outputs/encoder_megablocks_8_experts_6_GPUS_optimized_loops'], \
#         ['../outputs/encoder_sliced_1st_optim_32_experts_6_GPUS', '../outputs/encoder_deepspeed_32_experts_6_GPUS', '../outputs/encoder_megablocks_32_experts_6_GPUS', '../outputs/encoder_megablocks_32_experts_6_GPUS_optimized_loops'], \
#         ['../outputs/encoder_sliced_1st_optim_64_experts_6_GPUS', '../outputs/encoder_deepspeed_64_experts_6_GPUS', '../outputs/encoder_megablocks_64_experts_6_GPUS', '../outputs/encoder_megablocks_64_experts_6_GPUS_optimized_loops'], \
#         ['../outputs/encoder_sliced_1st_optim_128_experts_6_GPUS', '../outputs/encoder_deepspeed_128_experts_6_GPUS', '../outputs/encoder_megablocks_128_experts_6_GPUS', '../outputs/encoder_megablocks_128_experts_6_GPUS_optimized_loops'], \
#         ['../outputs/encoder_sliced_1st_optim_256_experts_6_GPUS', '../outputs/encoder_deepspeed_256_experts_6_GPUS', '../outputs/encoder_megablocks_256_experts_6_GPUS', '../outputs/encoder_megablocks_256_experts_6_GPUS_optimized_loops']]" 

# python plot_experiment.py \
#     --type "evolution" \
#     --evolution_values 8 16 32 64 128 256\
#     --evolution_objects  "Encoder-Decoder Sliced with optimization" "Deepspeed" "Encoder-Decoder Megablocks"\
#     --comparison_object "Deepspeed" \
#     --metric "Number of Experts" \
#     --comparisons "encoder-decoder_deepspeed_8_experts_6_GPUS" "encoder-decoder_deepspeed_16_experts_6_GPUS" "encoder-decoder_deepspeed_32_experts_6_GPUS" "encoder-decoder_deepspeed_64_experts_6_GPUS" "encoder-decoder_deepspeed_128_experts_6_GPUS" "encoder-decoder_deepspeed_256_experts_6_GPUS" \
#     --caching "load" "final" \
#     --dirs "[['../outputs/encoder-decoder_sliced_1st_optim_8_experts_6_GPUS', '../outputs/encoder-decoder_deepspeed_8_experts_6_GPUS', '../outputs/encoder-decoder_megablocks_8_experts_6_GPUS'], \
#         ['../outputs/encoder-decoder_sliced_1st_optim_16_experts_6_GPUS', '../outputs/encoder-decoder_deepspeed_16_experts_6_GPUS', '../outputs/encoder-decoder_megablocks_16_experts_6_GPUS'], \
#         ['../outputs/encoder-decoder_sliced_1st_optim_32_experts_6_GPUS', '../outputs/encoder-decoder_deepspeed_32_experts_6_GPUS', '../outputs/encoder-decoder_megablocks_32_experts_6_GPUS'],
#         ['../outputs/encoder-decoder_sliced_1st_optim_64_experts_6_GPUS', '../outputs/encoder-decoder_deepspeed_64_experts_6_GPUS', '../outputs/encoder-decoder_megablocks_64_experts_6_GPUS'], \
#         ['../outputs/encoder-decoder_sliced_1st_optim_128_experts_6_GPUS', '../outputs/encoder-decoder_deepspeed_128_experts_6_GPUS', '../outputs/encoder-decoder_megablocks_128_experts_6_GPUS'], \
#         ['../outputs/encoder-decoder_sliced_1st_optim_256_experts_6_GPUS', '../outputs/encoder-decoder_deepspeed_256_experts_6_GPUS', '../outputs/encoder-decoder_megablocks_256_experts_6_GPUS']]" 

# python plot_experiment.py \
#     --type "evolution" \
#     --evolution_values 1 2 4 6 8 \
#     --evolution_objects  "Encoder-Decoder Sliced with optimization" "Deepspeed" \
#     --comparison_object "Deepspeed" \
#     --metric "Number of GPUs" \
#     --comparisons "encoder-decoder_deepspeed_16_experts_1_gpus" "encoder-decoder_deepspeed_16_experts_2_gpus" "encoder_decoder_simple_deepspeed_16experts" "encoder-decoder_deepspeed_16_experts_6_gpus" "encoder-decoder_deepspeed_16_experts_8_gpus" \
#     --caching "load" "final" \
#     --dirs "[['../outputs/encoder-decoder_sliced_1st_optim_16_experts_1_gpus', '../outputs/encoder-decoder_deepspeed_16_experts_1_gpus'], \
#         ['../outputs/encoder-decoder_sliced_1st_optim_16_experts_2_gpus', '../outputs/encoder-decoder_deepspeed_16_experts_2_gpus'], \
#         ['../outputs/encoder-decoder_sliced_1st_optim_16_experts_0_parallel', '../outputs/encoder-decoder_simple_deepspeed_16_experts'],
#         ['../outputs/encoder-decoder_sliced_1st_optim_16_experts_6_gpus', '../outputs/encoder-decoder_deepspeed_16_experts_6_gpus'], \
#         ['../outputs/encoder-decoder_sliced_1st_optim_16_experts_8_gpus', '../outputs/encoder-decoder_deepspeed_16_experts_8_gpus']]" 

# python plot_experiment.py \
#     --type "evolution" \
#     --evolution_values 8 16 32 64 128 256\
#     --evolution_objects  "Deepspeed" "Encoder Megablocks" "Encoder Megablocks (Reduce-Scatter)" \
#     --comparison_object "Deepspeed" \
#     --metric "Number of Experts" \
#     --comparisons "Encoder DeepSpeed 8 Experts" "Encoder DeepSpeed 16 Experts" "Encoder DeepSpeed 32 Experts" "Encoder DeepSpeed 64 Experts" "Encoder DeepSpeed 128 Experts" "Encoder DeepSpeed 256 Experts" \
#     --caching "save"  \
#     --dirs "[['../outputs/encoder_deepspeed_8_exps_6_GPUS_new_code', '../outputs/encoder_mgblcks_8_exps_6_GPUS_new_code', '../outputs/encoder_mgblcks_8_exps_6_GPUS_reduce_scatter'], \
#         ['../outputs/encoder_deepspeed_16_exps_6_GPUS_new_code', '../outputs/encoder_mgblcks_16_exps_6_GPUS_new_code', '../outputs/encoder_mgblcks_16_exps_6_GPUS_reduce_scatter'], \
#         ['../outputs/encoder_deepspeed_32_exps_6_GPUS_new_code', '../outputs/encoder_mgblcks_32_exps_6_GPUS_new_code', '../outputs/encoder_mgblcks_32_exps_6_GPUS_reduce_scatter'], \
#         ['../outputs/encoder_deepspeed_64_exps_6_GPUS_new_code', '../outputs/encoder_mgblcks_64_exps_6_GPUS_new_code', '../outputs/encoder_mgblcks_64_exps_6_GPUS_reduce_scatter'], \
#         ['../outputs/encoder_deepspeed_128_exps_6_GPUS_new_code', '../outputs/encoder_mgblcks_128_exps_6_GPUS_new_code', '../outputs/encoder_mgblcks_128_exps_6_GPUS_reduce_scatter'], \
#         ['../outputs/encoder_deepspeed_256_exps_6_GPUS_new_code', '../outputs/encoder_mgblcks_256_exps_6_GPUS_new_code', '../outputs/encoder_mgblcks_256_exps_6_GPUS_reduce_scatter']]" 

# python plot_experiment.py \
#     --type "evolution" \
#     --evolution_values 8 16 32 64 128 256\
#     --evolution_objects  "Deepspeed" "Encoder-Decoder Megablocks" "Encoder-Decoder Megablocks (Reduce-Scatter)" \
#     --comparison_object "Deepspeed" \
#     --metric "Number of Experts" \
#     --comparisons "Encoder-Decoder DeepSpeed 8 Experts" "Encoder-Decoder DeepSpeed 16 Experts" "Encoder-Decoder DeepSpeed 32 Experts" "Encoder-Decoder DeepSpeed 64 Experts" "Encoder-Decoder DeepSpeed 128 Experts" "Encoder-Decoder DeepSpeed 256 Experts" \
#     --caching "load" "final" \
#     --dirs "[['../outputs/encoder-decoder_deepspeed_8_exps_6_GPUS_new_code', '../outputs/encoder-decoder_mgblcks_8_exps_6_GPUS_new_code', '../outputs/encoder-decoder_mgblcks_8_exps_6_GPUS_reduce_scatter'], \
#         ['../outputs/encoder-decoder_deepspeed_16_exps_6_GPUS_new_code', '../outputs/encoder-decoder_mgblcks_16_exps_6_GPUS_new_code', '../outputs/encoder-decoder_mgblcks_16_exps_6_GPUS_reduce_scatter'], \
#         ['../outputs/encoder-decoder_deepspeed_32_exps_6_GPUS_new_code', '../outputs/encoder-decoder_mgblcks_32_exps_6_GPUS_new_code', '../outputs/encoder-decoder_mgblcks_32_exps_6_GPUS_reduce_scatter'], \
#         ['../outputs/encoder-decoder_deepspeed_64_exps_6_GPUS_new_code', '../outputs/encoder-decoder_mgblcks_64_exps_6_GPUS_new_code', '../outputs/encoder-decoder_mgblcks_64_exps_6_GPUS_reduce_scatter'], \
#         ['../outputs/encoder-decoder_deepspeed_128_exps_6_GPUS_new_code', '../outputs/encoder-decoder_mgblcks_128_exps_6_GPUS_new_code', '../outputs/encoder-decoder_mgblcks_128_exps_6_GPUS_reduce_scatter'], \
#         ['../outputs/encoder-decoder_deepspeed_256_exps_6_GPUS_new_code', '../outputs/encoder-decoder_mgblcks_256_exps_6_GPUS_new_code', '../outputs/encoder-decoder_mgblcks_256_exps_6_GPUS_reduce_scatter']]" 

# python plot_experiment.py \
#     --type "evolution" \
#     --evolution_values 10 100 250 500\
#     --evolution_objects  "Deepspeed" "Encoder Megablocks" "Encoder Sliced" \
#     --comparison_object "Deepspeed" \
#     --metric "Batch Size" \
#     --comparisons "encoder_deepspeed_128_exps_6_GPUS_reduce_scatter_10_bs" "encoder_deepspeed_128_exps_6_GPUS_reduce_scatter_100_bs" "encoder_deepspeed_128_exps_6_GPUS_new_code" "encoder_deepspeed_128_exps_6_GPUS_reduce_scatter_500_bs" \
#     --caching "save"  \
#     --dirs "[['../outputs/encoder_deepspeed_128_exps_6_GPUS_reduce_scatter_10_bs', '../outputs/encoder_mgblcks_128_exps_6_GPUS_reduce_scatter_10_bs', '../outputs/encoder_sliced_1st_optim_128_exps_6_GPUS_reduce_scatter_10_bs'], \
#         ['../outputs/encoder_deepspeed_128_exps_6_GPUS_reduce_scatter_100_bs', '../outputs/encoder_mgblcks_128_exps_6_GPUS_reduce_scatter_100_bs', '../outputs/encoder_sliced_1st_optim_128_exps_6_GPUS_reduce_scatter_100_bs'], \
#         ['../outputs/encoder_deepspeed_128_exps_6_GPUS_new_code', '../outputs/encoder_mgblcks_128_exps_6_GPUS_reduce_scatter', '../outputs/encoder_sliced_1st_optim_128_exps_6_GPUS_reduce_scatter'], \
#         ['../outputs/encoder_deepspeed_128_exps_6_GPUS_reduce_scatter_500_bs', '../outputs/encoder_mgblcks_128_exps_6_GPUS_reduce_scatter_500_bs', '../outputs/encoder_sliced_1st_optim_128_exps_6_GPUS_reduce_scatter_500_bs']]"

# python plot_experiment.py \
#     --type "evolution" \
#     --evolution_values 10 100 250 500\
#     --evolution_objects  "Deepspeed" "Encoder-Decoder Megablocks" "Encoder-Decoder Sliced" \
#     --comparison_object "Deepspeed" \
#     --metric "Batch Size" \
#     --comparisons "encoder-decoder_deepspeed_128_exps_6_GPUS_reduce_scatter_10_bs" "encoder-decoder_deepspeed_128_exps_6_GPUS_reduce_scatter_100_bs" "encoder-decoder_deepspeed_128_exps_6_GPUS_new_code" "encoder-decoder_deepspeed_128_exps_6_GPUS_reduce_scatter_500_bs" \
#     --caching "load" "final"  \
#     --dirs "[['../outputs/encoder-decoder_deepspeed_128_exps_6_GPUS_reduce_scatter_10_bs', '../outputs/encoder-decoder_mgblcks_128_exps_6_GPUS_reduce_scatter_10_bs', '../outputs/encoder-decoder_sliced_1st_optim_128_exps_6_GPUS_reduce_scatter_10_bs'], \
#         ['../outputs/encoder-decoder_deepspeed_128_exps_6_GPUS_reduce_scatter_100_bs', '../outputs/encoder-decoder_mgblcks_128_exps_6_GPUS_reduce_scatter_100_bs', '../outputs/encoder-decoder_sliced_1st_optim_128_exps_6_GPUS_reduce_scatter_100_bs'], \
#         ['../outputs/encoder-decoder_deepspeed_128_exps_6_GPUS_new_code', '../outputs/encoder-decoder_mgblcks_128_exps_6_GPUS_reduce_scatter', '../outputs/encoder-decoder_sliced_1st_optim_128_exps_6_GPUS_reduce_scatter'], \
#         ['../outputs/encoder-decoder_deepspeed_128_exps_6_GPUS_reduce_scatter_500_bs', '../outputs/encoder-decoder_mgblcks_128_exps_6_GPUS_reduce_scatter_500_bs', '../outputs/encoder-decoder_sliced_1st_optim_128_exps_6_GPUS_reduce_scatter_500_bs']]"

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
#     --type "save_merge_csv" \
#     --evolution_type "num_experts" \
#     --evolution_label "Number of Experts" \
#     --rest "../outputs/mgblcks/encoder_mgblcks_8_experts_4_gpus    ../outputs/mgblcks/encoder_mgblcks_16_experts_4_gpus    ../outputs/mgblcks/encoder_mgblcks_32_experts_4_gpus    ../outputs/mgblcks/encoder_mgblcks_64_experts_4_gpus    ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus    ../outputs/mgblcks/encoder_mgblcks_256_experts_4_gpus \
#     ../outputs/deepspeed/encoder-decoder_deepspeed_8_experts_4_gpus    ../outputs/deepspeed/encoder-decoder_deepspeed_16_experts_4_gpus    ../outputs/deepspeed/encoder-decoder_deepspeed_32_experts_4_gpus    ../outputs/deepspeed/encoder-decoder_deepspeed_64_experts_4_gpus    ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus    ../outputs/deepspeed/encoder-decoder_deepspeed_256_experts_4_gpus    \
#     ../outputs/mgblcks/encoder-decoder_mgblcks_8_experts_4_gpus   ../outputs/mgblcks/encoder-decoder_mgblcks_16_experts_4_gpus    ../outputs/mgblcks/encoder-decoder_mgblcks_32_experts_4_gpus    ../outputs/mgblcks/encoder-decoder_mgblcks_64_experts_4_gpus    ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus    ../outputs/mgblcks/encoder-decoder_mgblcks_256_experts_4_gpus \
#     ../outputs/sliced/encoder_sliced_8_experts_4_gpus    ../outputs/sliced/encoder_sliced_16_experts_4_gpus    ../outputs/sliced/encoder_sliced_32_experts_4_gpus    ../outputs/sliced/encoder_sliced_64_experts_4_gpus    ../outputs/sliced/encoder_sliced_128_experts_4_gpus    ../outputs/sliced/encoder_sliced_256_experts_4_gpus    \
#     ../outputs/sliced/encoder-decoder_sliced_8_experts_4_gpus    ../outputs/sliced/encoder-decoder_sliced_16_experts_4_gpus    ../outputs/sliced/encoder-decoder_sliced_32_experts_4_gpus    ../outputs/sliced/encoder-decoder_sliced_64_experts_4_gpus    ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus    ../outputs/sliced/encoder-decoder_sliced_256_experts_4_gpus \
#     ../outputs/deepspeed/encoder_deepspeed_8_experts_4_gpus   ../outputs/deepspeed/encoder_deepspeed_16_experts_4_gpus    ../outputs/deepspeed/encoder_deepspeed_32_experts_4_gpus    ../outputs/deepspeed/encoder_deepspeed_64_experts_4_gpus    ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus    ../outputs/deepspeed/encoder_deepspeed_256_experts_4_gpus    "

# python plot_experiment.py \
#     --type "save_merge_csv" \
#     --evolution_type "batch_size" \
#     --evolution_label "Batch Size" \
#     --rest " ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_10_bs    ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_100_bs    ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus    ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_450_bs \
#     ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_10_bs    ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_100_bs    ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus    ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_450_bs   \
#     ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_10_bs    ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_100_bs    ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus    ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_450_bs \
#     ../outputs/sliced/encoder_sliced_128_experts_4_gpus_10_bs    ../outputs/sliced/encoder_sliced_128_experts_4_gpus_100_bs    ../outputs/sliced/encoder_sliced_128_experts_4_gpus    ../outputs/sliced/encoder_sliced_128_experts_4_gpus_450_bs \
#     ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_10_bs    ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_100_bs    ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus    ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_450_bs \
#    ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_10_bs    ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_100_bs    ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus    ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_450_bs"

# python plot_experiment.py \
#     --type "save_merge_csv" \
#     --evolution_type "num_experts_skew" \
#     --evolution_label "Number of Skewed Experts" \
#     --rest " ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_0.01_nes    ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_0.2_nes  ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_0.5_nes ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_0.6_nes ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus    ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_0.8_nes \
#     ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_0.01_nes    ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_0.2_nes  ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_0.5_nes  ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_0.6_nes ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_0.6_rs_90_cf   ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_0.8_nes   \
#     ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_0.01_nes    ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_0.2_nes  ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_0.5_nes ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_0.6_nes  ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus    ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_0.8_nes \
#     ../outputs/sliced/encoder_sliced_128_experts_4_gpus_0.01_nes    ../outputs/sliced/encoder_sliced_128_experts_4_gpus_0.2_nes  ../outputs/sliced/encoder_sliced_128_experts_4_gpus_0.5_nes ../outputs/sliced/encoder_sliced_128_experts_4_gpus_0.6_nes  ../outputs/sliced/encoder_sliced_128_experts_4_gpus    ../outputs/sliced/encoder_sliced_128_experts_4_gpus_0.8_nes \
#     ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_0.01_nes    ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_0.2_nes  ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_0.5_nes  ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_0.6_nes ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus    ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_0.8_nes \
#    ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_0.01_nes    ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_0.2_nes  ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_0.5_nes  ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_0.6_nes ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_0.6_rs_90_cf   ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_0.8_nes"

python plot_experiment.py \
    --type "save_merge_csv" \
    --evolution_type "router_skew" \
    --evolution_label "Router Skew" \
    --rest " ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_0.0_rs    ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_0.2_rs  ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_0.4_rs  ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus    ../outputs/mgblcks/encoder_mgblcks_128_experts_4_gpus_0.8_rs \
    ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_0.0_rs    ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_0.2_rs  ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_0.4_rs  ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus   ../outputs/deepspeed/encoder-decoder_deepspeed_128_experts_4_gpus_0.8_rs   \
    ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_0.0_rs    ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_0.2_rs  ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_0.4_rs  ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus    ../outputs/mgblcks/encoder-decoder_mgblcks_128_experts_4_gpus_0.8_rs \
    ../outputs/sliced/encoder_sliced_128_experts_4_gpus_0.0_rs    ../outputs/sliced/encoder_sliced_128_experts_4_gpus_0.2_rs  ../outputs/sliced/encoder_sliced_128_experts_4_gpus_0.4_rs  ../outputs/sliced/encoder_sliced_128_experts_4_gpus ../outputs/sliced/encoder_sliced_128_experts_4_gpus_0.8_rs \
    ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_0.0_rs    ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_0.2_rs  ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_0.4_rs  ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus    ../outputs/sliced/encoder-decoder_sliced_128_experts_4_gpus_0.8_rs \
   ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_0.0_rs_old    ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_0.2_rs_old  ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_0.4_rs_old  ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus   ../outputs/deepspeed/encoder_deepspeed_128_experts_4_gpus_0.8_rs_old"
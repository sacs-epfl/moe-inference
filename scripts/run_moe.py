import subprocess
import os
import math

os.chdir("../src")

# num_experts = [8, 16, 32, 64, 128, 256]
num_experts = [128]
model_type = ["encoder-decoder"]

expert_manager = ["sliced_fused_kernel", "manual_sliced"]
expert_name = {"simple": "deepspeed", "manual_sliced": "sliced", "sliced_fused_kernel": "mgblcks"}
expert_label = {"simple": "DeepSpeed", "sliced_fused_kernel": "MegaBlocks", "manual_sliced": "Sliced"}


BASE_PATH = "../outputs/{}/{}"
NUM_GPUS = 4

base_command = "python3 start_try.py"
base_args = {
    "model_type": None,
    "schedule": "deepspeed",
    "num_experts": None,
    "batch_size": 250,
    "seq_len": 120,
    "max_loaded_experts": None,
    "num_iters": 100,
    "expert_manager": None,
    "experiment_name": None,
    "num_parallel_experts_per_GPU": 0,
    "dataset": "bookcorpus",
    "profile": True,
    "label": None,
    "path": None,
    "num_experts_skew": None,
    "enable_router_skew": True,
    "random_router_skew": False,
    "router_skew": 0.6,
    "world": NUM_GPUS,
    "nvidia_profile": False
}

experiment_name = "prof_{}_{}_{}_experts_{}_gpus_2"
label_encoder = "'Encoder {} {} Experts'"
label_encoder_decoder = "'Encoder-Decoder {} {} Experts'"
for t in model_type:
    base_args["model_type"] = t
    for exp_mng in expert_manager:
        base_args["expert_manager"] = exp_mng

        for i, num_expert in enumerate(num_experts):
            if t == "encoder":
                base_args["label"] =  label_encoder.format(expert_label[exp_mng], num_expert)
            else:
                base_args["label"] = label_encoder_decoder.format(expert_label[exp_mng], num_expert)

            base_args["num_experts"] = num_expert
            base_args["max_loaded_experts"] = num_expert * 100
            base_args["num_experts_skew"] = math.ceil(num_expert * 0.1)

            name = experiment_name.format(t, expert_name[exp_mng], num_expert, NUM_GPUS)
            base_args["experiment_name"] = name

            path = BASE_PATH.format(expert_name[exp_mng], name)
            base_args["path"] = path


            command = f"{base_command} " + " ".join([f"--{key} {value}" for key, value in base_args.items()])

            print(f"Running experiment {name}")
            print (f"Running command: {command}")
            subprocess.run(command, shell=True)

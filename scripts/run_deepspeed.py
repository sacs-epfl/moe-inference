import subprocess
import math
import os

os.chdir("../src")

NUM_GPUS = 4

# num_experts = [8, 16, 32, 64, 128, 256]
num_experts = [128]
model_type = ["encoder", "encoder-decoder"]
num_experts_skew = [0.01, 0.2, 0.5, 0.6, 0.8] 
base_path = "../outputs/deepspeed/{}"

base_command = f"deepspeed --num_gpus {NUM_GPUS} start_deepspeed.py"
base_args = {
    "model_type": None,
    "num_experts": None,
    "num_iters": 100,
    "experiment_name": None,
    "dataset": "bookcorpus",
    "label": None,
    "router_skew": 0.6,
    "world_size": NUM_GPUS,
    "num_experts_skew": None,
    "random_router_skew": False,
    "enable_router_skew": True,
    "capacity_factor": None,
    "batch_size": 250,
    "seq_len": 120,
    "path": None
}

label_encoder = "'Encoder DeepSpeed {} Experts'"
label_encoder_decoder = "'Encoder-Decoder DeepSpeed {} Experts'"
experiment_name = "{}_deepspeed_{}_experts_{}_gpus_{}_nes"
for t in model_type:
    base_args["model_type"] = t
    for ne in num_experts:
        if t == "encoder":
            base_args["label"] =  label_encoder.format(ne)
        else:
            base_args["label"] = label_encoder_decoder.format(ne)

        base_args["num_experts"] = ne
        base_args["capacity_factor"] = min(ne, 90)

        for nes in num_experts_skew:
            base_args["num_experts_skew"] = math.ceil(ne * nes)
        
            name = experiment_name.format(t, ne, NUM_GPUS, nes)
            base_args["experiment_name"] = name
            base_args["path"] = base_path.format(name)


            command = f"{base_command} " + " ".join([f"--{key} {value}" for key, value in base_args.items()])

            print(f"Running experiment {name}")
            print (f"Running command: {command}")
            subprocess.run(command, shell=True)

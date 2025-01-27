import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp 
import sys
import os 
from datetime import datetime 
import time
import csv
import stat 
import json
import numpy as np
import signal
import argparse
import math
import random
import pynvml
from threading import Thread
import psutil
from tqdm import tqdm
from flexible_dataset import FlexibleDataset

import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from transformers import AutoTokenizer, SwitchTransformersEncoderModel, SwitchTransformersForConditionalGeneration, logging, AutoModel
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersDenseActDense
import warnings

from moe_layer_config import MoELayerConfig
from moe_layer import MoELayer
from setup_layers import replace_switch_moe_layer, parallelize_experts, save_latencies
from utils import debug
import pandas as pd


logging.set_verbosity_error()


def str2bool(s):
    return s.lower() in ["yes", "y", "true", "t"]

# Argparse
parser = argparse.ArgumentParser(
    prog="MoE workload generator",
    description="Spawns MoE model across GPUs and e2e iteration times",
)

parser.add_argument("-sl", "--seq_len", default=120, type=int)
parser.add_argument("-ni", "--num_iters", default=0, type=int)
parser.add_argument("-ns", "--num_samples", default=0, type=int, help="Number of samples per GPU")
parser.add_argument("-bs", "--batch_size", default=250, type=int, help="Batch size per GPU")
parser.add_argument("-w", "--world", default=torch.cuda.device_count(), type=int)
parser.add_argument("--warmup_rounds", default=3, type=int)
parser.add_argument("-p", "--port", default="1234", type=str)
parser.add_argument("-d", "--dataset", default="sst2", type=str)
parser.add_argument("-x", "--experiment", default="standard", type=str)
parser.add_argument("-e", "--num_experts", default=8, type=int)
parser.add_argument("-mt", "--model_type", default="encoder-decoder", type=str)
parser.add_argument("-pa", "--path", default="../outputs", type=str, help="Specify where to save path")
parser.add_argument("-em","--expert_manager", default="Sliced", type=str,choices=["Sliced", "MegaBlocks"], help="Which expert manager to use.")
parser.add_argument("-en", "--experiment_name", required=True, type=str, help="Name of the experiment, used for plotting metadata")
parser.add_argument("-lbl", "--label", default="", type=str, help="Label for the experiment")
parser.add_argument("-peGPU", "--num_parallel_experts_per_GPU", default=1, type=int, help="Number of parallel experts per GPU")
parser.add_argument("-prof", "--profile", default=False, type=str2bool, help="Profile the model")
parser.add_argument("-nvprof", "--nvidia_profile", default=False, type=str2bool, help="Profile the model with NVIDIA Nsight")
parser.add_argument("--num_experts_skew", default=3, type=int)
parser.add_argument("--random_router_skew", default=False, type=str2bool, help="Wether to enable random skewing in the router")
parser.add_argument("--router_skew", default=0.0, type=float, help="Value between 0 and 1")
parser.add_argument("--enable_router_skew", default=False, type=str2bool)

args = parser.parse_args()
print(args)

pynvml.nvmlInit()

if args.num_experts not in [8, 16, 32, 64, 128, 256]:
    print(f"There is no model with {args.num_experts} experts")
    exit(1)

if args.num_iters == 0 and args.num_samples == 0:
    print("You must either specify --num_iters or --num_samples")
    exit(1)

if args.num_parallel_experts_per_GPU == 0 and args.expert_manager == "Sliced":
    print("Number of parallel experts per GPU needs to be at least 1")
    exit(1)

if args.expert_manager == "MegaBlocks" and args.num_parallel_experts_per_GPU > 1:
    print("MegaBlocks does not support multiple parallel experts per GPU, ignoring --num_parallel_experts_per_GPU")


if args.num_iters != 0:
    DESIRED_DATASET_SIZE = args.num_iters * args.batch_size * args.world
else:
    DESIRED_DATASET_SIZE = args.num_samples

ROOT = f"{args.path}"

DEFAULT_CACHE_DIR = "/cache"
def get_cache():
    if "CACHE" in os.environ:
        return os.environ["CACHE"]
    else:
        print("Cache directory not set, using default ", DEFAULT_CACHE_DIR)
        return DEFAULT_CACHE_DIR

def setup(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port
    os.environ["HF_HOME"] = get_cache()
    os.environ["HF_DATASETS_CACHE"] = get_cache()
    dist.init_process_group("nccl", rank=rank, world_size=args.world)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def move_to_cuda_except_experts(model):
    for name, module in model.named_children():
        if name == 'experts':
            # We want to keep the experts on cpu
            continue
        elif list(module.children()):
            # If the module has children, recurse
            move_to_cuda_except_experts(module)
        else:
            # If it's a leaf module (no children) and not part of experts, move to CUDA
            module.cuda()

def percentage_of_total_params(model: nn.Module):
    total_params = 0 
    target_params = 0

    def count_params(module: nn.Module):
        nonlocal total_params
        nonlocal target_params

        debug(f"Module: {module.__class__.__name__}, Params: {sum(p.numel() for p in module.parameters(recurse=False))}")

        if type(module).__name__ == "SwitchTransformersSparseMLP":
            assert hasattr(module, "experts"), "Module does not have experts"
            for expert in module.experts.values():
                debug(f"\tModule: {expert.__class__.__name__}, Params: {sum(p.numel() for p in expert.parameters())}")
                inside_expert = sum(p.numel() for p in expert.parameters())
                total_params += inside_expert
                target_params += inside_expert

            for name, child in module.named_children():
                if name == "experts":
                    continue

                count_params(child)
        else:
            total_params += sum(p.numel() for p in module.parameters(recurse=False))
            for child in module.children():
                count_params(child)

    count_params(model)
    percentage = (target_params / total_params) * 100 if total_params > 0 else 0
    debug(f"Experts' percentage of total params: {percentage:.3f}%")
    debug(f"Total params: {total_params}")

def run_inference_workload(rank):
    try:
        mp.current_process().name = f'Worker-{rank}'
        setup(rank)

        tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8", cache_dir=get_cache())
        
        if args.model_type == "encoder-decoder":
            _class = AutoModel
        elif args.model_type == "encoder":
            _class = SwitchTransformersEncoderModel
        else:
            print("That model type is not yet implemented!")
            exit(1)

        config = MoELayerConfig(
            expert_manager=args.expert_manager,
            num_experts=args.num_experts,
            num_parallel_experts_per_GPU=args.num_parallel_experts_per_GPU,
            profile=args.profile,
            nv_profile=args.nvidia_profile,
            enable_router_skew=args.enable_router_skew,
            router_skew=args.router_skew,
            random_router_skew=args.random_router_skew,
            num_experts_skew=args.num_experts_skew,
        )
        
        model = _class.from_pretrained(
            f"google/switch-base-{args.num_experts}", 
            # expert_capacity=11718, 
            cache_dir=get_cache(),
        )

        replace_switch_moe_layer(model, MoELayer, config)
        move_to_cuda_except_experts(model)
        parallelize_experts(model)

        debug(model)

        datasets.enable_caching()
        flexible_dataset = FlexibleDataset(
            args.dataset,
            tokenizer=tokenizer, 
            model=model, 
            seq_len=args.seq_len,
            num_samples=DESIRED_DATASET_SIZE,
        )

        sampler = DistributedSampler(
            flexible_dataset, 
            num_replicas=args.world, 
            rank=rank, 
            shuffle=True, 
            seed=49
        )

        loader = DataLoader(flexible_dataset, sampler=sampler, batch_size=args.batch_size)

        model.eval()

        if args.experiment == "standard":
            run_start, run_end = run_standard_experiment(model, tokenizer, loader, f"{ROOT}/{rank}")
            if rank == 0:
                save_run_info(ROOT, run_start, run_end)
        else:
            print(f"That experiment, {args.experiment}, is not yet implemented")
            exit(1)

    except KeyboardInterrupt:
        print(f"Worker {rank} received KeyboardInterrupt, shutting down...")
    finally:
        cleanup()

def run_standard_experiment(model, tokenizer, loader, path):
    latencies = []

    with torch.no_grad():
        # WARMUP
        itr = 0
        for batch in loader:
            batch = {k: v.cuda() for k, v in batch.items()}

            if args.model_type == "encoder-decoder":
                model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"],
                )
            else:
                model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
            itr += 1
            if itr == args.warmup_rounds:
                break

        run_start = time.time()
        # RUN ACTUAL EXPERIMENT
        for batch in tqdm(loader):
            start = time.time()
            batch = {k: v.cuda() for k, v in batch.items()}
            if args.model_type == "encoder-decoder":
                model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"],
                )
            else:
                model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
            end = time.time()
            latencies.append(end - start)
        run_end = time.time()

    
    create_save_dir_if_not_exist(path)

    save_latencies(model, path, args.warmup_rounds)

    file_path = f"{path}/e2e.csv"
    with open(file_path, "w") as f:
        fieldnames = ["iteration", "latency (s)"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, latency in enumerate(latencies):
            writer.writerow({"iteration": idx, "latency (s)": latency})
    
    return run_start, run_end

def warmup(model, loader):
    NUM_WARMUP_ROUNDS = 3

    itr = 0
    for batch in loader:
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = model(**batch)
        itr += 1
        if itr == NUM_WARMUP_ROUNDS:
            break

def create_save_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_run_info(path, run_start, run_end):
     with open(f"{path}/data.json", "w") as f:
        json.dump({
            "name": args.experiment_name,
            "label": args.label if args.label else args.experiment_name,
            "batch_size": args.batch_size,
            "world_size": args.world,
            "dataset": args.dataset,
            "seq_len": args.seq_len,
            "num_iters": args.num_iters,
            "num_samples": args.num_samples,
            "port": args.port,
            "experiment": args.experiment,
            "num_experts": args.num_experts,
            "model_type": args.model_type,
            "expert_manager": args.expert_manager,
            "num_parallel_experts_per_GPU": args.num_parallel_experts_per_GPU,
            "profile": args.profile,
            "start": run_start,
            "end": run_end,
            "enable_router_skew": args.enable_router_skew,
            "router_skew": args.router_skew,
            "random_router_skew": args.random_router_skew,
            "num_experts_skew": args.num_experts_skew,
        }, f)

def signal_handler(sig, frame):
    print("Main process received Ctrl+C! Terminating all child processes...")
    for child in mp.active_children():
         print(f"Terminating child process PID: {child.pid}")
         child.terminate()
    sys.exit(0)

def fetch_metrics(stop_event, output_list):
    handles = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(args.world)]

    while not stop_event.is_set():
        output_list.append({
            "timestamp": time.time(),
            "gpu_util": [pynvml.nvmlDeviceGetUtilizationRates(handle).gpu for handle in handles],
            "gpu_mem_used": [pynvml.nvmlDeviceGetMemoryInfo(handle).used for handle in handles],
            "cpu_util": psutil.cpu_percent(interval=None),
            "cpu_mem_used": psutil.virtual_memory().used,
        })

        time.sleep(1)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    metrics = []
    stop_event = mp.Event()

    metric_thread = Thread(target=fetch_metrics, args=(stop_event, metrics))
    metric_thread.start()

    mp.spawn(run_inference_workload, nprocs=args.world, join=True)

    stop_event.set()
    metric_thread.join()

    df = pd.DataFrame(metrics)
    df.to_csv(f"{ROOT}/stats.csv")

    pynvml.nvmlShutdown()
    

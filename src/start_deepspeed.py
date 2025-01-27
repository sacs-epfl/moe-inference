# https://proceedings.mlr.press/v162/rajbhandari22a/rajbhandari22a.pdf
# https://github.com/microsoft/DeepSpeed

import torch
import torch.nn as nn
import pynvml
import psutil
import torch.multiprocessing as mp 
from threading import Thread
import sys
import os 
import time
import csv
import json
import pandas as pd
from tqdm import tqdm

from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel
from transformers import SwitchTransformersEncoderModel, SwitchTransformersForConditionalGeneration, logging

from flexible_dataset import FlexibleDataset
import argparse 

from utils import TimedModule, get_timing_modules
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersBlock, SwitchTransformersLayerSelfAttention, SwitchTransformersLayerCrossAttention
from transformers.models.switch_transformers import SwitchTransformersTop1Router

from router import Router

import deepspeed
from deepspeed.moe.sharded_moe import top1gating, top2gating, topkgating
from deepspeed.utils.timer import SynchronizedWallClockTimer
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple, Union
from utils import debug
import torch.distributed as dist

def str2bool(s):
        return s.lower() in ["yes", "y", "true", "t"]

parser = argparse.ArgumentParser(
    prog="Run inference on DeepSpeed-MoE inference engine",
)
parser.add_argument("--dataset", default="sst2", type=str)
parser.add_argument("--num_samples", default=0, type=int, help="Number of total samples across all GPUs")
parser.add_argument("--batch_size", default=250, type=int, help="Batch size per GPU")
parser.add_argument("--seq_len", default=120, type=int)
parser.add_argument("-ni", "--num_iters", default=0, type=int)
parser.add_argument("--path", default="../outputs/out", type=str, help="Specify where to save path")
parser.add_argument("--num_experts", default=8, type=int, help="Number of experts we want to match dense model to")
parser.add_argument("--warmup_rounds", default=3, type=int)
parser.add_argument("--local_rank", default=0, type=int) 
parser.add_argument("--world_size", default=8, type=int)
parser.add_argument("--capacity_factor", default=10.0, type=float)
parser.add_argument("--enable_router_skew", default=False, type=str2bool)
parser.add_argument("--router_skew", default=0.0, type=float, help="Value between 0 and 1")
parser.add_argument("-lbl", "--label", default="", type=str, help="Label for the experiment")
parser.add_argument("-en", "--experiment_name", required=True, type=str, help="Name of the experiment, used for plotting metadata")
parser.add_argument("--random_router_skew", default=False, type=str2bool, help="Wether to enable random skewing in the router")
parser.add_argument("-mt", "--model_type", default="encoder-decoder", type=str)
parser.add_argument("--num_experts_skew", default=3, type=int)
args = parser.parse_args()


if args.num_iters != 0:
    DESIRED_DATASET_SIZE = args.num_iters * args.batch_size * args.world_size
else:
    DESIRED_DATASET_SIZE = args.num_samples 

############# GLOBAL AFFAIRS ################
pynvml.nvmlInit()
#############################################
DEFAULT_CACHE_DIR = "/cache"
def get_cache():
    if "CACHE" in os.environ:
        return os.environ["CACHE"]
    else:
        print("Cache directory not set, using default ", DEFAULT_CACHE_DIR)
        return DEFAULT_CACHE_DIR

def setup():
    os.environ["HF_HOME"] = get_cache()
    os.environ["HF_DATASETS_CACHE"] = get_cache()
    os.environ["TRITON_HOME"] = "/.triton"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    deepspeed.init_distributed()

    torch.cuda.set_device(args.local_rank)

def run_inference_workload():
    setup()

    model_name = f"google/switch-base-{args.num_experts}"
    if args.model_type == "encoder-decoder":
        _class = AutoModel
    elif args.model_type == "encoder":
        _class = SwitchTransformersEncoderModel
    else:
        print("That model type is not yet implemented!")
        exit(1)
    model = _class.from_pretrained(model_name, cache_dir=get_cache())

    class MLPWrapper(nn.Module):
        def __init__(self, child):
            super().__init__()
            self.child = child
        
        def forward(self, x):
            x = self.child(x)
            return x[0], (x[1], x[2])

    class TopKGate(nn.Module):
        """Gate module which implements Top2Gating as described in Gshard_.
        ::

            gate = TopKGate(model_dim, num_experts)
            l_aux, combine_weights, dispatch_mask = gate(input)

        .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

        Args:
            model_dim (int):
                size of model embedding dimension
            num_experts (int):
                number of experts in model
        """

        wg: torch.nn.Linear

        def __init__(self,
                    model_dim: int,
                    num_experts: int,
                    k: int = 1,
                    capacity_factor: float = 1.0,
                    eval_capacity_factor: float = 1.0,
                    min_capacity: int = 8,
                    noisy_gate_policy: Optional[str] = None,
                    drop_tokens: bool = True,
                    use_rts: bool = True,
                    ep_group: Union[torch.distributed.ProcessGroup, None] = None,
                    top2_2nd_expert_sampling: bool = True) -> None:
            super().__init__()

            self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
            self.ep_group = ep_group
            self.k = k
            self.capacity_factor = capacity_factor
            self.eval_capacity_factor = eval_capacity_factor
            self.min_capacity = min_capacity
            self.noisy_gate_policy = noisy_gate_policy
            self.timers = SynchronizedWallClockTimer()
            self.wall_clock_breakdown = False
            self.gate_time = 0.0
            self.drop_tokens = drop_tokens
            self.use_rts = use_rts
            self.top2_2nd_expert_sampling = top2_2nd_expert_sampling
            # Here is the update by adding my own router I can change the logits
            # I want to do this before the chaos that comes after which I cannot rewrite
            self.router = Router(args.num_experts, skew=args.router_skew, num_expert_skew=args.num_experts_skew, enable_random=args.random_router_skew) 
            self.enable_router_skew = args.enable_router_skew

        def _set_ep_group(self, ep_group):
            assert self.ep_group is None, f'Attempting to override an existing ep_group'
            self.ep_group = ep_group

        def forward(self,
                    input: torch.Tensor,
                    used_token: torch.Tensor = None,
                    use_tutel: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore

            if self.wall_clock_breakdown:
                self.timers(TOPK_GATE_TIMER).start()

            input_fp32 = input.float()
            # input jittering
            if self.noisy_gate_policy == 'Jitter' and self.training:
                input_fp32 = multiplicative_jitter(input_fp32, device=input.device)
            
            if self.enable_router_skew:
                 _, _, logits = self.router(input_fp32)
            else:
                logits = torch.nn.functional.linear(input_fp32, weight=self.wg.weight.float(), bias=None)
           

            if self.k == 1:
                gate_output = top1gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                        self.min_capacity, used_token, self.noisy_gate_policy if self.training else None,
                                        self.drop_tokens, self.use_rts, self.ep_group, use_tutel)

            elif self.k == 2:
                gate_output = top2gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                        self.min_capacity, self.drop_tokens, self.ep_group, self.top2_2nd_expert_sampling)
            else:
                gate_output = topkgating(logits, self.k,
                                        self.capacity_factor if self.training else self.eval_capacity_factor,
                                        self.min_capacity, self.drop_tokens, self.ep_group)

            if self.wall_clock_breakdown:
                self.timers(TOPK_GATE_TIMER).stop()
                self.gate_time = self.timers(TOPK_GATE_TIMER).elapsed(reset=False)
            
            return gate_output

    # Update to add DeepspeedMoE to it
    def add_deepspeed_moe_model(model):
        _add_deepspeed_moe_model(model, [-1], [-1], False)

    def _add_deepspeed_moe_model(model, layer_encoder, layer_decoder, is_decoder=False):
        is_decoder = is_decoder
        layer_encoder = layer_encoder

        if type(model).__name__ == "SwitchTransformersBlock":
            assert hasattr(model, "is_decoder"), "SwitchTransformersBlock does not have is_decoder attribute"
            is_decoder = getattr(model, "is_decoder")
            if is_decoder:
                layer_decoder[0] += 1
            else:
                layer_encoder[0] += 1

        for name, module in model.named_children():
            if type(module).__name__ == "SwitchTransformersSparseMLP":
                assert hasattr(module, "experts"), "SwitchTransformersSparseMLP does not have experts attribute"
                assert hasattr(module, "router"), "SwitchTransformersSparseMLP does not have router attribute"

                router = getattr(module, "router")
                experts = getattr(module, "experts")

                if type(experts) == nn.ModuleDict:
                    experts = list(experts.values())
                
                num_experts_per_gpu = args.num_experts // args.world_size
                experts = experts[args.local_rank*num_experts_per_gpu:(args.local_rank+1)*num_experts_per_gpu]
                layer_idx = layer_decoder[0] if is_decoder else layer_encoder[0]

                new = deepspeed.moe.layer.MoE(
                    hidden_size=768,
                    expert=experts[0],
                    num_experts=args.num_experts,
                    ep_size=args.world_size,
                    k=1,
                    eval_capacity_factor=args.capacity_factor,
                    use_tutel=True,
                    top2_2nd_expert_sampling=False,
                    enable_expert_tensor_parallelism=True,
                    use_rts=False,
                )

                setattr(new.deepspeed_moe, "gate", 
                TopKGate(768, args.num_experts, 1, 1.0, args.capacity_factor, 8, None, True, False, None, False)
                )

                with torch.no_grad():
                    new.deepspeed_moe.gate.wg.weight.copy_(router.classifier.weight)
                    for i in range(len(experts)):
                        new.deepspeed_moe.experts.deepspeed_experts[i].wi.weight.copy_(experts[i].wi.weight)
                        new.deepspeed_moe.experts.deepspeed_experts[i].wo.weight.copy_(experts[i].wo.weight)
                    
                setattr(model, name, TimedModule(MLPWrapper(new), idx=layer_idx, is_decoder=is_decoder))
            
            else:
                _add_deepspeed_moe_model(module, layer_encoder, layer_decoder, is_decoder)

    add_deepspeed_moe_model(model)

    model.eval()
    model.cuda()


    ds_engine = deepspeed.init_inference(
        model,
        dtype=torch.float,
        replace_with_kernel_inject=False,
        moe={
            "enabled": True,
            "ep_size": args.world_size, 
            "moe_experts": [args.num_experts],
        },
        quant={
            "enabled": False,
        },
    )


    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=get_cache())

    flexible_dataset = FlexibleDataset(
        args.dataset, 
        tokenizer, 
        model, 
        seq_len=args.seq_len,
        num_samples=DESIRED_DATASET_SIZE,
    )
    sampler = DistributedSampler(
        flexible_dataset, 
        num_replicas=args.world_size, 
        rank=args.local_rank, 
        shuffle=True, 
        seed=49
    )
    loader = DataLoader(
        flexible_dataset, 
        sampler=sampler, 
        batch_size=args.batch_size
    )


    latencies, run_start, run_end = run_standard_experiment(ds_engine, loader)

    path = f"{args.path}/{args.local_rank}"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    ############# E2E #######################
    file_path = f"{path}/e2e.csv"
    with open(file_path, "w") as f:
        fieldnames = ["iteration", "Latency (s)"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, latency in enumerate(latencies):
            writer.writerow({
                "iteration": idx, 
                "Latency (s)": latency,
            })
    
    ############# LAYER #######################
    for timing_module in get_timing_modules([], model):
        latencies = timing_module.get_latencies()[args.warmup_rounds:]
        file_path = f"{path}/moe_l{timing_module.idx}"
        if timing_module.is_decoder:
            file_path += "_decode"
        file_path += ".csv"
        with open(file_path, "w") as f:
            fieldnames = ["iteration", "latency (ms)"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx, latency in enumerate(latencies):
                writer.writerow({
                    "iteration": idx,
                    "latency (ms)": latency,
                })
    
    ############# META #######################
    if args.local_rank == 0:
        run_info = vars(args).copy()
        with open(f"{args.path}/data.json", "w") as f:
            json.dump({ "experiment_name": args.experiment_name, "label": args.label, "start": run_start, "end": run_end, **run_info, "world_size": args.world_size}, f, indent=4)

def run_standard_experiment(ds_engine, loader):
    latencies = []
    
    with torch.no_grad():
        # WARMUP
        itr = 0
        for batch in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            if args.model_type == "encoder-decoder":
                ds_engine(
                    input_ids=batch["input_ids"], 
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"],
                )
            else:
                ds_engine(
                    input_ids=batch["input_ids"], 
                    attention_mask=batch["attention_mask"],
                )
            itr += 1
            if itr == args.warmup_rounds:
                break

        # RUN ACTUAL EXPERIMENT
        run_start = time.time()
        for batch in tqdm(loader):
            start = time.time()
            batch = {k: v.cuda() for k, v in batch.items()}
            if args.model_type == "encoder-decoder":
                ds_engine(
                    input_ids=batch["input_ids"], 
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"],
                )
            else:
                ds_engine(
                    input_ids=batch["input_ids"], 
                    attention_mask=batch["attention_mask"],
                )

            end = time.time()
            latencies.append(end-start)
        run_end = time.time()
    
    return latencies, run_start, run_end

def fetch_metrics(stop_event, output_list):
    handles = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(args.world_size)]

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
    metrics = []
    stop_event = mp.Event()

    metric_thread = Thread(target=fetch_metrics, args=(stop_event, metrics))
    metric_thread.start()

    run_inference_workload()

    stop_event.set()
    metric_thread.join()

    df = pd.DataFrame(metrics)
    df.to_csv(f"{args.path}/stats.csv")

    print("All done :)")

    pynvml.nvmlShutdown()
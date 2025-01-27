import csv

import torch.nn as nn
import torch.distributed as dist
import torch

from utils import debug
from expert_manager import ManualSlicedExpertManager, SlicedExpertManagerFusedKernel
from moe_layer_config import MoELayerConfig

import megablocks.ops as ops
from router import Router






class MoELayer(nn.Module):
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """

    def __init__(self, config: MoELayerConfig, old_experts, old_router=None, layer_idx=None, is_decoder=False):
        super().__init__()
        # Step 1: Get the correct router according to its class
        self.config = config

        if config.enable_router_skew:
            self.router = Router(num_experts=config.num_experts, skew=config.router_skew, num_expert_skew=config.num_experts_skew, enable_random=config.random_router_skew)
        else:
            self.router = old_router if old_router is not None else config.router_class(config)

        self.num_gpus = dist.get_world_size()
        self.rank = dist.get_rank()
        self.num_experts = config.num_experts
        self.layer_idx = layer_idx
        self.is_decoder = is_decoder

        if config.expert_manager == "Sliced":
            self.expert_manager = ManualSlicedExpertManager(old_experts, config)
        elif config.expert_manager == "MegaBlocks":
            self.expert_manager = SlicedExpertManagerFusedKernel(old_experts, config)
        else:
            raise ValueError(f"Expert manager {config.expert_manager} not supported")
        
        self.num_iters  = 0
        

    def get_tensor_size(self, tensor):
        return tensor.numel() * tensor.element_size()
    
            
    def forward(self, hidden_states):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        result = self.timed_forward(hidden_states)

        end.record()
        torch.cuda.synchronize()
        self.expert_manager.e2e_layer_time.append(start.elapsed_time(end))

        return result

        
    
    @torch.no_grad()
    def timed_forward(self, hidden_states):
        r"""
        Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:

        1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
        and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
        hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).

        2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
        expert the corresponding hidden states.

        """
        if self.expert_manager.nv_profile and self.num_iters == 3:
            torch.cuda.cudart().cudaProfilerStart()
        
        self.expert_manager.maybe_nv_range("Forward")

        self.expert_manager.maybe_start_profiling("Before Metadata and 1st Data")

        # Step 1: Get the router_mask from the router as wel as the probabilities
        router_mask, router_probs, router_logits = self.router(hidden_states)
        expert_index = torch.argmax(router_mask, dim=-1)
        # router_mask has dim (batch_size, seq_len, num_experts)
        # Entry will be a 1 on which expert to work on for the specific token
        # at specific sequence index on specific sample, rest will be 0

        # The routers introduced might not always map all the tokens, to a router, which means that some hidden states
        # can be unchanged from one layer to another. That is why the hidden states are cloned before updating only the seleced ones.

        # hidden_states shape: (batch_size, seq_len, d_model)
        next_states = hidden_states.clone()


        router_mask = router_mask.bool()
        batch_size, seq_len, num_experts = router_mask.shape

        self.expert_manager.maybe_nv_range("Router mask")
        #shape of experts_inputs entry: (num_tokens, d_model)
        experts_inputs = [hidden_states[router_mask[:,:,idx]] for idx in range(num_experts)]
        self.expert_manager.maybe_nv_range_pop()

        # I (GPU x) send to GPU y the chunk (along the features dim) that GPU y needs to process, not the whole tensor
        # This way, I only send to each GPU the bare minimum of data that it needs to process
        # The alternative would be to send all of my shard of the data and then each GPU selects the chunk it needs

        # However, to be able to use an all_to_all, I need to concatenate inputs (that I will send to GPU y) from different experts into a single tensor

        # That means that I first need to send metadata so that the receiver can then split the tensor into the inputs from each expert again
        # because each expert might have a different number of tokens assigned to it

        # The metadata sent by GPU x is the same for all other GPUs, because each GPU receives all of GPU x's samples 
        # (a chunk in the features dim, but the same amount of samples per expert)
        # if self.expert_manager.is_first_split_row_wise():
        debug(f"Experts inputs shapes: {[experts_input.shape for experts_input in experts_inputs]}", rank=self.rank)
        self.expert_manager.maybe_nv_range("First cat")
        to_send = torch.cat(experts_inputs, dim=0)
        self.expert_manager.maybe_nv_range_pop()

        sizes_send = torch.tensor([experts_input.shape[0] for experts_input in experts_inputs], dtype=torch.int, device="cuda", requires_grad=False)
        sizes_recv = torch.empty(self.num_gpus, self.num_experts, dtype=torch.int, requires_grad=False, device="cuda")

        self.expert_manager.tot_num_bytes_sent.append(self.get_tensor_size(sizes_send) * (self.num_gpus - 1))
        self.expert_manager.tot_num_bytes_recv.append(self.get_tensor_size(sizes_send) * (self.num_gpus - 1))

        self.expert_manager.maybe_nv_range("Metadata all gather")
        self.expert_manager.maybe_end_profiling("Before Metadata and 1st Data")
        self.expert_manager.maybe_start_profiling("Metadata and 1st Data")

        dist.all_gather_into_tensor(sizes_recv, sizes_send)

        self.expert_manager.maybe_nv_range_pop()

        to_recv = [
            torch.empty(sizes_recv[i].sum(), hidden_states.shape[2], device="cuda", requires_grad=False) 
                for i in range(self.num_gpus)
        ]

        self.expert_manager.maybe_nv_range("First data all gather")

        dist.all_gather(to_recv, to_send)

        self.expert_manager.maybe_end_profiling("Metadata and 1st Data")
        self.expert_manager.maybe_nv_range_pop()
        self.expert_manager.maybe_start_profiling("Before 2nd Data")

        self.expert_manager.tot_num_bytes_sent[-1] += self.get_tensor_size(to_send) * (self.num_gpus - 1)
        self.expert_manager.tot_num_bytes_recv[-1] += sum(self.get_tensor_size(t) for i, t in enumerate(to_recv) if i != self.rank)


        self.expert_manager.maybe_nv_range("Contiguous")

        sizes_recv_t = sizes_recv.T.contiguous()

        self.expert_manager.maybe_nv_range_pop()
        self.expert_manager.maybe_nv_range("First loop")

        # first, split the tensor from each gpu into inputs per expert from each gpu again
        all_inputs = [[None for _ in range(num_experts)] for _ in range(self.num_gpus)]

        #these indices are the start indices of the input for each expert coming from each GPU
        start_idx_per_GPU = ops.exclusive_cumsum(sizes_recv, 1).cpu()

        # these indexes are used to split the output from each expert into the inputs from each GPU again
        start_idx_per_expert = ops.exclusive_cumsum(sizes_recv_t, 1).cpu()

        del sizes_recv_t


        for gpu in range(self.num_gpus):
            start_idx_gpu = start_idx_per_GPU[gpu]
            for expert_idx in range(num_experts - 1):
                all_inputs[gpu][expert_idx] = to_recv[gpu][start_idx_gpu[expert_idx]:start_idx_gpu[expert_idx + 1], :]

            all_inputs[gpu][num_experts - 1] = to_recv[gpu][start_idx_gpu[num_experts - 1]:, :]

        self.expert_manager.maybe_nv_range_pop()
        self.expert_manager.maybe_nv_range("Second cat")

        # to run the experts, concatenate the inputs of the same expert from different gpus
        workload = [torch.cat([all_inputs[i][j] for i in range(self.num_gpus)], dim=0) for j in range(num_experts)]
        del all_inputs
        self.expert_manager.maybe_nv_range_pop()

        # run the experts
        self.expert_manager.maybe_nv_range("Execute job")
        workload = self.expert_manager.execute_job(workload)
        self.expert_manager.maybe_nv_range_pop()

        
        self.expert_manager.maybe_nv_range("Second loop")

        # get only the ouput correspondent to the tokens that were assigned to each GPU (because data parallel)
        tokens_outputs_per_owner = [[None for _ in range(self.num_experts)] for _ in range(self.num_gpus)]
        for expert_idx, expert_output in enumerate(workload):
            for gpu in range(self.num_gpus):
                start = start_idx_per_expert[expert_idx, gpu]
                if gpu == self.num_gpus - 1:
                    end = expert_output.shape[0]
                else:
                    end = start_idx_per_expert[expert_idx, gpu + 1]
                
                tokens_outputs_per_owner[gpu][expert_idx] = expert_output[start:end, :]
        
        self.expert_manager.maybe_nv_range_pop()
        del start_idx_per_expert

        debug(f"Tokens outputs per owner shapes: {[[t.shape for t in tokens_owner] for tokens_owner in tokens_outputs_per_owner]}", rank=self.rank)

        self.expert_manager.maybe_nv_range("Third cat")
        to_send = [
            torch.cat([tokens_outputs_per_owner[gpu][expert_idx] for expert_idx in range(num_experts)], dim=0)
            for gpu in range(self.num_gpus)
        ]
        self.expert_manager.maybe_nv_range_pop()

        to_recv = torch.empty(sizes_recv[self.rank].sum(), hidden_states.shape[2], device="cuda", requires_grad=False)

        self.expert_manager.maybe_nv_range("Data ReduceScatter")
        self.expert_manager.maybe_end_profiling("Before 2nd Data")
        self.expert_manager.maybe_start_profiling("2nd Data")

        dist.reduce_scatter(to_recv, to_send, op=dist.ReduceOp.SUM)

        self.expert_manager.maybe_nv_range_pop()
        self.expert_manager.maybe_end_profiling("2nd Data")
        self.expert_manager.maybe_start_profiling("After 2nd Data")

        self.expert_manager.tot_num_bytes_sent[-1] += sum(self.get_tensor_size(t) for i, t in enumerate(to_send) if i != self.rank)
        self.expert_manager.tot_num_bytes_recv[-1] += self.get_tensor_size(to_recv)
        
        my_start_idx = start_idx_per_GPU[self.rank]

        self.expert_manager.maybe_nv_range("Third loop")

        copy_results(num_experts, next_states, router_mask, to_recv, my_start_idx)
        self.expert_manager.maybe_nv_range_pop()

        if self.expert_manager.nv_profile:
            if self.num_iters == 3:
                torch.cuda.cudart().cudaProfilerStop()
            else:
                self.num_iters += 1

        self.expert_manager.maybe_nv_range_pop()

        hidden_states = router_probs * next_states
        self.expert_manager.maybe_end_profiling("After 2nd Data")
        return hidden_states, (router_logits, expert_index)

    
    @torch.jit.ignore
    def expert_save_latencies(self, DIR="", warmup=0):
        path = f"{DIR}/moe_l{self.layer_idx}"
        if self.is_decoder:
            path += "_decode"

        with open(f"{path}.csv", "w") as f:
            fieldnames = ["iteration", "latency (ms)", "total number of bytes sent", "total number of bytes recv"]

            if self.expert_manager.profile:
                fieldnames += list(self.expert_manager.get_timer_names())
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(len(self.expert_manager.tot_num_bytes_sent[warmup:])):
                dic = {
                    "iteration": i,
                    "latency (ms)": self.expert_manager.e2e_layer_time[i],
                    "total number of bytes sent": self.expert_manager.tot_num_bytes_sent[i],
                    "total number of bytes recv": self.expert_manager.tot_num_bytes_recv[i]
                }

                if self.expert_manager.profile:
                    for timer_name in self.expert_manager.get_timer_names():
                        dic[timer_name] = self.expert_manager.results[timer_name][i]

                writer.writerow(dic)



@torch.jit.script
def copy_results(num_experts:int, new_hidden_states:torch.Tensor, router_mask:torch.Tensor, to_copy:torch.Tensor, indexes:torch.Tensor):
    for expert_idx in range(num_experts - 1):
        new_hidden_states[router_mask[:,:,expert_idx]] = to_copy[indexes[expert_idx]:indexes[expert_idx + 1], :]
    
    new_hidden_states[router_mask[:,:,num_experts - 1]] = to_copy[indexes[num_experts - 1]:, :]

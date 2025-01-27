
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import copy
import gc

from utils import debug, promote_scalar, ChunkType
from moe_layer_config import MoELayerConfig

import stk
import stk.backend.triton_kernels
import stk.ops
import megablocks.ops as ops

from abc import ABC, abstractmethod
from collections import defaultdict



############################################ EXPERT CLASS ############################################
class ChunkedLinear(nn.Module):
    def __init__(self, in_features, out_features, type: ChunkType):
        super(ChunkedLinear, self).__init__()
        self.num_chunks =  dist.get_world_size()
        self.type = type
        self.rank = dist.get_rank()

        if type == ChunkType.ROW:
            self.sizes = [in_features // self.num_chunks for _ in range(self.num_chunks)]
            rest = in_features % self.num_chunks
            if rest != 0:
                for i in range(rest):
                    self.sizes[i] += 1

            self.linear = nn.Linear(self.sizes[self.rank], out_features, bias=False)
            
        else:
            self.sizes = [out_features // self.num_chunks for _ in range(self.num_chunks)]
            rest = out_features % self.num_chunks
            if rest != 0:
                for i in range(rest):
                    self.sizes[i] += 1
            self.linear = nn.Linear(in_features, self.sizes[self.rank], bias=False)
        
        curr = 0
        for i in range(len(self.sizes)):
            start = curr
            end = curr + self.sizes[i]
            self.sizes[i] = (start, end, self.sizes[i])
            curr = end
    
    def forward(self, hidden_states):
        return self.linear(hidden_states) 


class Expert(nn.Module):
    def __init__(self, config: MoELayerConfig):
        super().__init__()

        # column and then row because that means that we do not need to chunk the input tensor
        # also, since the out_features of wi and the in_features of wo are the same, and the chunking algorithm is deterministic,
        # the shapes will already be prepared for the 2nd matrix multiplication

        # these variables are access in the code via self.wi and self.wo after `expert_parallelize` is called
        # we need to keep the old ones to be able to load the model
        self.wi = ChunkedLinear(config.d_model, config.d_ff, ChunkType.COLUMN)
        self.wo = ChunkedLinear(config.d_ff, config.d_model, ChunkType.ROW)

        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = nn.ReLU()

        self.rank = dist.get_rank()

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
    

################################ EXPERT MANAGER CLASSES ################################

class ExpertManager(ABC, nn.Module):
    def __init__(self, old_experts: nn.ModuleDict, config: MoELayerConfig):
        super().__init__()
        self.experts = nn.ModuleDict()
        for i in range(len(old_experts)):
            self.experts[f"expert_{i}"] = Expert(config)
        

        self.num_experts = len(old_experts)
        self.rank = dist.get_rank()
        self.num_gpus = dist.get_world_size()

        self.tot_num_bytes_sent = []
        self.tot_num_bytes_recv = []
        self.e2e_layer_time = []

        self.start_timers = defaultdict(lambda: torch.cuda.Event(enable_timing=True)) if config.profile else None
        self.end_timers = defaultdict(lambda: torch.cuda.Event(enable_timing=True)) if config.profile else None
        self.results = defaultdict(list) if config.profile else None



        self.profile = config.profile
        self.nv_profile = config.nv_profile

        self.execute_job_e2e_time = [] if self.profile else None

        self.num_iters = 0
        self.copy_expert_weights(old_experts)


    @abstractmethod
    def expert_parallelise(self):
        pass
    
    @abstractmethod
    def execute_job(self, workload: list):
        pass

    def forward(self, workload: list):
        return self.execute_job(workload)

    def maybe_start_profiling(self, name):
        if self.profile:
            self.start_timers[name].record()
    
    def maybe_end_profiling(self, name):
        if self.profile:
            end = self.end_timers[name]
            end.record()
            end.synchronize()
            self.results[name].append(self.start_timers[name].elapsed_time(end))
    
    def get_timer_names(self):
        return self.results.keys()
    
    def maybe_nv_range(self, range):
        if self.nv_profile and self.num_iters == 3:
            torch.cuda.nvtx.range_push(range)
    
    def maybe_nv_range_pop(self):
        if self.nv_profile and self.num_iters == 3:
            torch.cuda.nvtx.range_pop()

    def _copy_expert_weights(self, old_expert, new_expert, wi_type, wo_type):
        start_wi, end_wi, _ = new_expert.wi.sizes[self.rank]
        start_wo, end_wo, _ = new_expert.wo.sizes[self.rank]
        if wi_type == ChunkType.ROW:
            new_expert.wi.linear.weight.copy_(old_expert.wi.weight[:, start_wi:end_wi])
        else:
            new_expert.wi.linear.weight.copy_(old_expert.wi.weight[start_wi:end_wi, :])
        
        if wo_type == ChunkType.ROW:
            new_expert.wo.linear.weight.copy_(old_expert.wo.weight[:, start_wo:end_wo])
        else:
            new_expert.wo.linear.weight.copy_(old_expert.wo.weight[start_wo:end_wo, :])
    

    def copy_expert_weights(self, old_experts):
        with torch.no_grad():
            # For each expert, we load the chunk that is assigned to the current GPU
            # For now, I assume that we can load the entire experts into memory
            for i, old_expert in enumerate(old_experts.values()):
                new_expert = self.experts[f"expert_{i}"]
                wi_type = new_expert.wi.type
                wo_type = new_expert.wo.type
                
                self._copy_expert_weights(old_expert, new_expert, wi_type, wo_type)



class ManualSlicedExpertManager(ExpertManager):
    def __init__(self, experts: nn.ModuleDict, config: MoELayerConfig):
        super().__init__(experts, config)

        self.streams = [torch.cuda.Stream() for _ in range(config.num_parallel_experts_per_GPU)] if config.num_parallel_experts_per_GPU > 1 else None
        self.num_streams = len(self.streams) if self.streams is not None else 0


    def expert_parallelise(self):
        with torch.no_grad():
            self.experts = self.experts.cuda()

    def is_first_split_row_wise(self):
        return self.experts["expert_0"].wi.type == ChunkType.ROW
    
    def get_expert_wi_chunk_indexes(self, rank):
        return self.experts[f"expert_0"].wi.sizes[rank]


    def execute_job(self, workload: list):
        # For each expert, keep only the tokens that are going to run in the current gpu
        used_experts = []
        for expert_number, work in enumerate(workload):
            if workload[expert_number].size(dim=0) == 0:
                continue
            used_experts.append(expert_number)
            if self.num_streams == 0:
                workload[expert_number] = self.experts[f"expert_{expert_number}"](work) 
            else:
                with torch.cuda.stream(self.streams[expert_number % self.num_streams]):
                    workload[expert_number] = self.experts[f"expert_{expert_number}"](work)
        
        if self.num_streams != 0:
            for stream in self.streams:
                stream.synchronize()

        return workload



    
class SlicedExpertManagerFusedKernel(ExpertManager):
    def __init__(self, experts: nn.ModuleDict, config: MoELayerConfig):
        super().__init__(experts, config)

        self.blocking = 128 # blocking factor for sparse matrix multiplication kernels
        self.ffn_hidden_size = self.experts["expert_0"].wo.linear.weight.shape[1]
        self.hidden_size = self.experts["expert_0"].wi.linear.weight.shape[1]

        debug(f"Expert wi shape: {self.experts['expert_0'].wi.linear.weight.shape}")
        debug(f"Expert wo shape: {self.experts['expert_0'].wo.linear.weight.shape}")
        debug(f"FFN hidden size: {self.ffn_hidden_size}")
        debug(f"Hidden size: {self.hidden_size}")

        # Calculate the number of bits needed to represent the column indices
        # in the intermediate sparse matrix.
        max_column_index = ((self.ffn_hidden_size * self.num_experts) // self.blocking)
        self.transpose_sort_end_bit = max(
            int(np.ceil(np.log2(max_column_index))),
            1,
        )

        self.fused_wi = nn.Parameter(
            torch.empty(
                self.hidden_size, 
                self.ffn_hidden_size * self.num_experts,
                dtype=torch.float16,
                requires_grad=False
                )
            )

        self.fused_wo = nn.Parameter(
            torch.empty(
                self.ffn_hidden_size * self.num_experts, 
                self.hidden_size,
                dtype=torch.float16,
                requires_grad=False
                )
            )
        
        debug(f"Fused wi shape: {self.fused_wi.shape}")
        debug(f"Fused wo shape: {self.fused_wo.shape}")

        self.act = nn.ReLU()
        
        self.expert_wi_chunk_indexes = copy.deepcopy(self.experts["expert_0"].wi.sizes)

    @torch.jit.ignore
    def expert_parallelise(self):
        with torch.no_grad():

            self.fused_wi.data = torch.cat([expert.wi.linear.weight.data.T for expert in self.experts.values()], dim=1).cuda()
            self.fused_wo.data = torch.cat([expert.wo.linear.weight.data.T for expert in self.experts.values()], dim=0).cuda()

            # from here on we do not need to keep the experts
            delattr(self, "experts")
            gc.collect()
                
    def get_expert_wi_chunk_indexes(self, rank):
        return self.expert_wi_chunk_indexes[rank]

    def execute_job(self, workload: list):
        debug(f"Workload shapes: {[work.shape for work in workload]}")
        
        x, indices, bin_ids, bins, padded_bins, tokens_per_expert = self.indices_and_padded_bins(workload)

        x = ops.padded_gather(
            x,
            indices,
            bin_ids,
            bins,
            padded_bins,
            1
        )

        with torch.no_grad():
            topo = self.topology(x, padded_bins)
            
            # apply wi
            debug(f"X size: { x.size()} Device: {x.device}")
            debug(f"fused_wi size: {self.fused_wi.size()} Device: {self.fused_wi.device}")
            debug(f"topo size: {topo.size()} Device: {topo.device}")
            x = stk.ops.sdd(x, self.fused_wi, topo)

            # apply the activation function
            # not applying dropout because I assume this is inference
            x = stk.Matrix(
                x.size(),
                self.act(x.data),
                x.row_indices,
                x.column_indices,
                x.offsets,
                x.column_indices_t,
                x.offsets_t,
                x.block_offsets_t,
            )

            x = stk.ops.dsd(x, self.fused_wo)
        
        #TODO: not sure if this works, because this function takes the expert wieghts and I do not know what it does with it. I am going to pass a tensor with 1's and assume it performs scaling
        expert_weights = torch.ones((indices.size(0),1), device=x.device)
        debug(f"X size: {x.size()}")
        debug(f"Indices size: {indices.size()}")
        x = ops.padded_scatter(
            x,
            indices,
            bin_ids,
            expert_weights,
            bins,
            padded_bins,
            1
        )

        workload = torch.split(x, tokens_per_expert.tolist(), dim=0)

        return workload

    # function taken from ParallelDroplessMLP from MegaBlocks, modified to work with our setup
    def sparse_transpose(self, size, row_indices, column_indices, offsets):
        block_columns = size[1] // self.blocking

        # Sort row indices by column indices to get the transposed matrix's
        # column indices.
        #
        # NOTE: Our sort operation uses the same width indices as the input values.
        # To avoid overflow when we have large activation matrices we cast to
        # 32-bit before sorting.
        _, gather_indices = ops.sort(
            column_indices.int(),
            self.transpose_sort_end_bit,
        )

        # There are a constant number of blocks in every row of the sparse matrix.
        # A blocks offset is:
        #
        # row_index * blocks_per_row + column_index % blocks_per_row
        #
        # Once we have the block offsets ordered for transposition we can divide
        # by blocks_per_row to get the transposed column indices.
        column_indices_t = row_indices.gather(0, gather_indices.long())
        block_offsets_t = gather_indices.int()

        zero = torch.zeros((1,), dtype=torch.int32, device=row_indices.device)
        nnz_per_column = ops.histogram(column_indices, block_columns)
        nnz_per_column = ops.inclusive_cumsum(nnz_per_column, 0)
        if nnz_per_column.dim() == 0:
            # This addresses an edge case when ffn_hidden_size is equal to self.blocking.
            nnz_per_column = nnz_per_column.unsqueeze(0)
        offsets_t = torch.cat([zero, nnz_per_column])
        return column_indices_t, offsets_t, block_offsets_t
    

    # custom function to get the indices, bin_ids, bins, padded_bins, tokens_per_expert to further re utilize mega blocks code
    def indices_and_padded_bins(self, workload):
        indices = torch.arange(0, sum([work.shape[0] for work in workload]), device=workload[0].device)

        values = torch.tensor([i for i in range(self.num_experts)], device=workload[0].device)
        tokens_per_expert = torch.tensor([work.shape[0] for work in workload], device=workload[0].device, dtype=torch.int32)
        bin_ids = torch.repeat_interleave(values, tokens_per_expert)
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = promote_scalar(bins)

        padded_tokens_per_expert = ops.round_up(tokens_per_expert, self.blocking)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        padded_bins = promote_scalar(padded_bins)

        # TODO: assumes that the expert matrices shape (the one which is sharded) is divisible by the number of GPUS
        x = torch.cat(workload, dim=0)

        return x, indices, bin_ids, bins, padded_bins, tokens_per_expert
    
    # function taken from ParallelDroplessMLP from MegaBlocks, modified to work with our setup
    def topology(self, x, padded_bins):
        padded_tokens, _ = x.size()
        assert padded_tokens % self.blocking == 0
        if self.ffn_hidden_size % self.blocking != 0:
            raise ValueError(
                f'The ffn_hidden_size {self.ffn_hidden_size} must be divisible by ' +
                f'the block size {self.blocking}. Please update your configuration.',
            )

        # Offsets for the sparse matrix. All rows have the
        # same number of nonzero blocks dictated by the
        # dimensionality of a single expert.
        block_rows = padded_tokens // self.blocking
        blocks_per_row = self.ffn_hidden_size // self.blocking
        offsets = torch.arange(
            0,
            block_rows * blocks_per_row + 1,
            blocks_per_row,
            dtype=torch.int32,
            device=x.device,
        )

        # Indices for the sparse matrix. The indices for
        # the intermediate matrix are dynamic depending
        # on the mapping of tokens to experts.
        column_indices = ops.topology(
            padded_bins,
            self.blocking,
            block_rows,
            blocks_per_row,
        )

        # TODO(tgale): This is unused. Remove the need for this in stk.
        # For now, use meta init to save the device memory.
        data = torch.empty(
            column_indices.numel(),
            self.blocking,
            self.blocking,
            dtype=torch.float16,
            device='meta',
        )
        debug("Data shape: ", data.shape)
        shape = (
            padded_tokens,
            self.ffn_hidden_size * self.num_experts,
        )
        row_indices = stk.ops.row_indices(shape, data, offsets, column_indices)
        column_indices_t, offsets_t, block_offsets_t = self.sparse_transpose(
            shape,
            row_indices,
            column_indices,
            offsets,
        )

        return stk.Matrix(
            shape,
            data,
            row_indices,
            column_indices,
            offsets,
            column_indices_t,
            offsets_t,
            block_offsets_t,
        )
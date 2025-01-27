
from transformers.configuration_utils import PretrainedConfig
from transformers.models.switch_transformers import SwitchTransformersTop1Router

class MoELayerConfig(PretrainedConfig):
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}
    model_type = "switch_transformers"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        d_model=768,
        d_kv=64,
        d_ff=2048,
        expert_capacity=64,
        num_layers=12,
        num_sparse_encoder_layers=3,
        num_decoder_layers=12,
        num_sparse_decoder_layers=3,
        num_heads=12,
        num_experts=8,
        ep_size=1,
        router_bias=False,
        router_jitter_noise=0.01,
        router_dtype="float32",
        router_ignore_padding_tokens=False,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        router_z_loss_coef=0.001,
        router_aux_loss_coef=0.001,
        initializer_factor=1.0,
        dense_act_fn="relu",
        is_encoder_decoder=True,
        add_router_probs=False,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        router_class=SwitchTransformersTop1Router,
        expert_manager="manual_sliced",
        num_parallel_experts_per_GPU=0,
        profile=False,
        nv_profile=False,
        enable_router_skew=False,
        router_skew=0.0,
        random_router_skew=False,
        num_experts_skew=1,
        **kwargs,
    ):
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.expert_capacity = expert_capacity
        self.num_layers = num_layers
        self.num_sparse_encoder_layers = num_sparse_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_sparse_decoder_layers = num_sparse_decoder_layers
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.router_bias = router_bias
        self.router_jitter_noise = router_jitter_noise
        self.router_dtype = router_dtype
        self.router_ignore_padding_tokens = router_ignore_padding_tokens
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.router_z_loss_coef = router_z_loss_coef
        self.router_aux_loss_coef = router_aux_loss_coef
        self.initializer_factor = initializer_factor
        self.dense_act_fn = dense_act_fn
        self.is_encoder_decoder = is_encoder_decoder
        self.add_router_probs = add_router_probs
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.expert_manager = expert_manager
        self.num_parallel_experts_per_GPU = num_parallel_experts_per_GPU
        self.profile = profile
        self.nv_profile = nv_profile
        self.router_class = router_class
        self.enable_router_skew = enable_router_skew
        self.router_skew = router_skew
        self.random_router_skew = random_router_skew
        self.num_experts_skew = num_experts_skew


        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
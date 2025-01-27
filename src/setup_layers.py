def replace_switch_moe_layer(model, new_layer_class, config):
    _replace_switch_moe_layer(model, new_layer_class, config, [-1], [-1], False)

def _replace_switch_moe_layer(model, new_layer_class, config, layer_encoder, layer_decoder, is_decoder=False):
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
            
            layer_idx = layer_decoder[0] if is_decoder else layer_encoder[0]
            new_layer = new_layer_class(config, experts, router, layer_idx, is_decoder)

            setattr(model, name, new_layer)
        
        else:
            _replace_switch_moe_layer(module, new_layer_class, config, layer_encoder, layer_decoder, is_decoder)


def parallelize_experts(model):
    for name, module in model.named_children():
        if name == "expert_manager":
            module.expert_parallelise()
        else:
            parallelize_experts(module)


def save_latencies(model, path, warmup):
    for module in model.children():
        if type(module).__name__ == "MoELayer":
            module.expert_save_latencies(path, warmup)
        else:
            save_latencies(module, path, warmup)
def attention_head_only_training(model):
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the attention heads
    for block in model.blocks:
        for name, param in block.named_parameters():
            if "attn" in name:
                param.requires_grad = True
    return model




def decoupled_weight_decay(weight_decay_dict, model):

    param_groups = []
    #param_group_names = []
    for name, parameter in model.named_parameters():
        if 'head' in name:
            wd =  weight_decay_dict['head']
            parameter.requires_grad = True
        else:
            wd = weight_decay_dict['base']

        if parameter.requires_grad:
            param_groups.append({'params': [parameter], 'weight_decay': wd})
            #param_group_names.append(name)

    return param_groups

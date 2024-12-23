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









def initilize_CosineAnnealingWarmRestarts(optimizer,scheduler,warmup_scheduler,T_0=1, T_mult=2, eta_min=0.000001,
                                          last_epoch=-1,multiplier = 0.75,warmup_end_factor=1.0, 
                                            warmup_total_iters=300
                                         
                                         
                                         ):
    
    class CosineAnnealingWarmRestarts_2(scheduler):
        def __init__(self,optimizer,T_0=1, T_mult=2, eta_min=0.000001,last_epoch=-1,multiplier = 0.75,
                    warmup_end_factor=1.0, 
                    warmup_total_iters=300
                    
                    ):
            super().__init__(optimizer,
                        T_0 = T_0, 
                        T_mult = T_mult, 
                        eta_min = eta_min,
                        last_epoch = last_epoch)
            self.multiplier = multiplier
            
            #self.base_lrs[0] = base_lrs
    
            self.warmup = warmup_scheduler(
                        optimizer,
                        start_factor=eta_min,
                        end_factor=warmup_end_factor ,
                        total_iters=warmup_total_iters,
                    )
            
            
            
        
        def step(self,epoch=20,iteration=1,Batch_count=300):
    
            super().step(epoch + iteration / Batch_count)

            if (iteration == Batch_count-1)and (epoch+1 + self.T_0) % self.T_i ==0 :
                #print('decrease')

                self.base_lrs[0] = self.base_lrs[0] * self.multiplier
            
        def warmup_step(self):
            self.warmup.step()
    
    
    train_scheduler = CosineAnnealingWarmRestarts_2(optimizer,
                                T_0 = T_0, 
                                T_mult = T_mult, 
                                eta_min = eta_min,
                                last_epoch = last_epoch,multiplier = multiplier,
                                                   
                               warmup_end_factor = warmup_end_factor,
                                warmup_total_iters = warmup_total_iters
                                                   
                                                   )

    return train_scheduler

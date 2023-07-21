"""
If you are contributing a new agent, implement your 
agent initialization & predict functions here. 

Next, (optionally) update your agent's name in agent/__init__.py.
"""

import numpy as np
import torchvision.transforms as T
from .Agent import Agent
from RMDT.RMDTTransformer import mem_transformer
import torch

NUM_JOINTS = 7

class CollaboratorAgent(Agent):
    def __init__(
            self, n_token, n_layer, n_head, d_model, d_head, d_inner, dropout, dropatt, MEM_LEN, ext_len, 
            tie_weight, num_mem_tokens, mem_at_end, device, learning_rate, betas, context_length, sections):
        
        self.model = mem_transformer.MemTransformerLM(n_token=n_token, n_layer=n_layer, n_head=n_head, 
                                                      d_model=d_model, d_head=d_head, d_inner=d_inner, 
                                                      dropout=dropout, dropatt=dropatt, mem_len=MEM_LEN, 
                                                      ext_len=ext_len, tie_weight=tie_weight, 
                                                      num_mem_tokens=num_mem_tokens, mem_at_end=mem_at_end)
        self.device = device
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, betas=betas)
        self.raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.EFFECTIVE_SIZE_BLOCKS = context_length * sections
        self.BLOCKS_CONTEXT = context_length

    def predict(self, observation: dict):
        # TODO: replace with your predict function
        return np.random.choice(self.delta_range, NUM_JOINTS)
    
    def train(self, batch):
        losses = []
        self.model.train()
        memory = None
        mem_tokens=None
        for block_part in range(self.EFFECTIVE_SIZE_BLOCKS//self.BLOCKS_CONTEXT):
                    
            from_idx = block_part*(self.BLOCKS_CONTEXT)
            to_idx = (block_part+1)*(self.BLOCKS_CONTEXT)
            x1 = batch["observations"][:, from_idx:to_idx, :].to(self.device)
            y1 = batch["actions"][:, from_idx:to_idx, :].to(self.device)
            r1 = batch["rewards"][:, from_idx:to_idx, :].to(self.device)
            
            if mem_tokens is not None:
                mem_tokens = mem_tokens.detach()
            elif self.raw_model.mem_tokens is not None:
                mem_tokens = self.raw_model.mem_tokens.repeat(1, x1.shape[0], 1)
                    
            with torch.set_grad_enabled(True):
                if memory is not None:
                    res = self.model(x1, y1, r1, y1, None, *memory, mem_tokens=mem_tokens) # timesteps = None
                else:
                    res = self.model(x1, y1, r1, y1, None, mem_tokens=mem_tokens)
                memory = res[0][2:]
                logits, loss = res[0][0], res[0][1]
                
                mem_tokens = res[1]
                
                loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())
        return loss.mean()


def _init_agent_from_config(config, device='cpu'):
    # TODO: replace with your init_agent_from_config function
    agent = CollaboratorAgent(
        n_token=config.n_token, 
        n_layer=config.n_layer, 
        n_head=config.n_head, 
        d_model=config.d_model, 
        d_head=config.d_head, 
        d_inner=config.d_inner, 
        dropout=config.dropout, 
        dropatt=config.dropatt, 
        MEM_LEN=config.MEM_LEN, 
        ext_len=config.ext_len, 
        tie_weight=config.tie_weight, 
        num_mem_tokens=config.num_mem_tokens, 
        mem_at_end=config.mem_at_end,
        device=device,
        learning_rate=config.learning_rate,
        betas=config.betas,
        context_length=config.context_length,
        sections=config.sections,
        )
    transforms = lambda x: x
    return agent, transforms
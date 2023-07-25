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
import os

NUM_JOINTS = 7

class CollaboratorAgent(Agent):
    def __init__(
            self, n_layer, n_head, d_model, d_head, d_inner, dropout, dropatt, MEM_LEN, ext_len, 
            num_mem_tokens, mem_at_end, device, learning_rate, context_length, sections, grad_norm_clip):
        
        self.model = mem_transformer.MemTransformerLM(n_layer=n_layer, n_head=n_head, 
                                                      d_model=d_model, d_head=d_head, d_inner=d_inner, 
                                                      dropout=dropout, dropatt=dropatt, mem_len=MEM_LEN, 
                                                      ext_len=ext_len, num_mem_tokens=num_mem_tokens, max_ep_len=context_length,
                                                      mem_at_end=mem_at_end, STATE_DIM=7, ACTION_DIM=7, IMG_DIM=2048)
        self.device = device
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.EFFECTIVE_SIZE_BLOCKS = context_length * sections
        self.BLOCKS_CONTEXT = context_length
        self.learning_rate = learning_rate
        self.grad_norm_clip = grad_norm_clip

    def predict(self, observation: dict):
        # TODO: replace with your predict function
        return np.random.choice(self.delta_range, NUM_JOINTS)
    
    def train(self, batch):
        losses = []
        self.model.train()
        memory = None
        mem_tokens=None
        length = batch["length"].detach().cpu().numpy()
        for block_part in range(self.EFFECTIVE_SIZE_BLOCKS//self.BLOCKS_CONTEXT):
                    
            from_idx = block_part*(self.BLOCKS_CONTEXT)
            to_idx = (block_part+1)*(self.BLOCKS_CONTEXT)
            x1 = batch["observations"][:, from_idx:to_idx, :].to(self.device)
            y1 = batch["actions"][:, from_idx:to_idx, :].to(self.device)
            r1 = batch["rewards"][:, from_idx:to_idx, :].to(self.device)
            i1 = batch["embeddings"][:, from_idx:to_idx, :].to(self.device)
            lengths = [l if l < self.BLOCKS_CONTEXT*4 else self.BLOCKS_CONTEXT*4 for l in length]
            length = [max(0, (l - self.BLOCKS_CONTEXT*4)) for l in length]
            
            if mem_tokens is not None:
                mem_tokens = mem_tokens.detach()
            elif self.raw_model.mem_tokens is not None:
                mem_tokens = self.raw_model.mem_tokens.repeat(1, x1.shape[0], 1)
                    
            with torch.set_grad_enabled(True):
                if memory is not None:
                    res = self.model(x1, y1, r1, y1, None, *memory, mem_tokens=mem_tokens, img=i1, length=lengths) # timesteps = None
                else:
                    res = self.model(x1, y1, r1, y1, None, mem_tokens=mem_tokens, img=i1, length=lengths)
                memory = res[0][2:]
                logits, loss = res[0][0], res[0][1]
                
                mem_tokens = res[1]
                
                loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())
            if sum(length) == 0:
                break
        self.model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        self.optimizer.step()
        return loss
    
    def eval(self, batch):
        losses = []
        self.model.eval()
        memory = None
        mem_tokens=None
        length = batch["length"].detach().cpu().numpy()
        for block_part in range(self.EFFECTIVE_SIZE_BLOCKS//self.BLOCKS_CONTEXT):
                    
            from_idx = block_part*(self.BLOCKS_CONTEXT)
            to_idx = (block_part+1)*(self.BLOCKS_CONTEXT)
            x1 = batch["observations"][:, from_idx:to_idx, :].to(self.device)
            y1 = batch["actions"][:, from_idx:to_idx, :].to(self.device)
            r1 = batch["rewards"][:, from_idx:to_idx, :].to(self.device)
            i1 = batch["embeddings"][:, from_idx:to_idx, :].to(self.device)
            lengths = [l if l < self.BLOCKS_CONTEXT*4 else self.BLOCKS_CONTEXT*4 for l in length]
            length = [max(0, (l - self.BLOCKS_CONTEXT*4)) for l in length]
            
            if mem_tokens is not None:
                mem_tokens = mem_tokens.detach()
            elif self.raw_model.mem_tokens is not None:
                mem_tokens = self.raw_model.mem_tokens.repeat(1, x1.shape[0], 1)
                    
            with torch.no_grad():
                if memory is not None:
                    res = self.model(x1, y1, r1, y1, None, *memory, mem_tokens=mem_tokens, img=i1, length=lengths) # timesteps = None
                else:
                    res = self.model(x1, y1, r1, y1, None, mem_tokens=mem_tokens, img=i1, length=lengths)
                memory = res[0][2:]
                logits, loss = res[0][0], res[0][1]
                
                mem_tokens = res[1]
                
                loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())
            if sum(length) == 0:
                break
        return loss
    
    def save(self, foldername, filename='Agent.pth', epoch=0):
        state = {'epoch': epoch,
                 'optimizer': self.optimizer.state_dict(),
                 }
        state["model"] = self.model.state_dict()
        torch.save(state, os.path.join(foldername, filename))


def _init_agent_from_config(config, device='cpu'):
    # TODO: replace with your init_agent_from_config function
    agent = CollaboratorAgent(
        n_layer=config.n_layer, 
        n_head=config.n_head, 
        d_model=config.d_model, 
        d_head=config.d_head, 
        d_inner=config.d_inner, 
        dropout=config.dropout, 
        dropatt=config.dropatt, 
        MEM_LEN=config.MEM_LEN, 
        ext_len=config.ext_len, 
        num_mem_tokens=config.num_mem_tokens, 
        mem_at_end=config.mem_at_end,
        device=device,
        learning_rate=config.learning_rate,
        context_length=config.context_length,
        sections=config.sections,
        grad_norm_clip=config.grad_norm_clip,
        )
    transforms = lambda x: x
    return agent, transforms
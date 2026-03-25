"""
GPT-2 Model Architecture — built from scratch.

Implements the full GPT-2 (124M) transformer:
  CausalAttention → MLP → Block → GPT

Supports:
  - Flash Attention via PyTorch's scaled_dot_product_attention
  - Weight tying between token embeddings and output head
  - HuggingFace weight loading for all GPT-2 model sizes
  - Fused AdamW optimizer with parameter-group weight decay
"""

import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

class CausalAttention(nn.Module):
    """Multi-head causal self-attention with Flash Attention."""
    
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key,query,value for all heads but in a single matrix
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANO_GPT_SCALE_INIT = 1;
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # # mask for attention to void future context                               # not required since we are using Flash Attention it applies it internally
        # self.register_buffer("bias",torch.tril(torch.ones(config.block_size, config.block_size)).
        #                     view(1,1,config.block_size, config.block_size)) # we added two more dim cause our attention att dim is (B, nh, T, T)


    def forward(self, x):
        B, T, C = x.size() # batch_size, sequence_length, channels (or) n_embd
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2) # each(B, T, C)
        # make the attention head
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        # calculate the attention
        # att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        # att = F.softmax(att, dim=-1) 
        # y = att @ v  #(B, nh, T, T)  x (B, nh, T, hs) -> (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v,is_causal=True) # Flash Attention indicate torch.compile to perform kernal fusion on Attention
        y = y.transpose(1,2).contiguous().view(B, T, C) # back to original input dim
        # output projection to communication between multi-head
        y = self.c_proj(y)
        return y 

class MLP(nn.Module):
    """Position-wise feed-forward network with GELU activation."""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANO_GPT_SCALE_INIT = 1
    def forward(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    """Transformer block: LayerNorm → Attention → LayerNorm → MLP (pre-norm)."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    """Hyperparameters for the GPT model architecture."""
    block_size: int = 1024 # max-block size
    vocab_size: int = 50304 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # num of layers 
    n_head: int = 12 # num of heads
    n_embd: int = 768 # embedding dim

class GPT(nn.Module):
    """GPT-2 language model with configurable size."""
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
            h = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
            ln_f = nn.LayerNorm(self.config.n_embd)
        ))
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANO_GPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0, std = 0.02)
    
    def forward(self, idx,targets=None):
        # idx shape is (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot find sequence length{T}, block size is only {self.config.block_size}"
        # forward token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        # add both
        x = tok_emb + pos_emb # here broadcasting happens
        # forward through layers or block
        for layer in self.transformer.h:
            x = layer(x)

        # pass through final layer_norm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy( logits.view(-1, logits.size(-1)), targets.view(-1),ignore_index=-1)
        return logits, loss
            
    @classmethod
    def from_pretrained(cls, model_type):
        """Loading pretrained GPT-2 model from hugging face"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # remove the buffer which is not a param

        # init a huggingface transformer model
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = hf_model.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    #Optimizer Configuration
    def optimizer_configuration(self, weight_decay, lr, device, master_process):

        # selecting the parameter that requires grad
        param_dict = {pn:p for pn, p in self.named_parameters() if p.requires_grad}

        # select the parameters that has 2 or more dim
        decay_param = [p for pn, p in param_dict.items() if p.dim() >=2]
        nodecay_param = [p for pn, p in param_dict.items() if p.dim() < 2]

        # creating optim group
        optim_groups = [
            {'params':decay_param, 'weight_decay':weight_decay},
            {'params': nodecay_param, 'weight_decay':0}
        ]

        num_decay_param = sum(p.numel() for p in decay_param)
        num_nodecay_param = sum(p.numel() for p in nodecay_param)
        if master_process:
          print(f"Number of decay param tensors : {len(decay_param)} | Number of decay parameters {num_decay_param}")
          print(f"Number of non-decay param tensors : {len(nodecay_param)} | Number of non-decay parameters {num_nodecay_param}")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters 
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr = lr, betas=(0.9, 0.95), eps=1e-8, fused = use_fused)
        return optimizer


import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from copy import deepcopy
from typing import Optional
import re

def mm(a, b, masking=True):
    # Canonically, a is the weight tensor, b is the mask tensor
    if a is not None and b is not None and masking:
        return a * b
    elif a is not None:
        return a
    else:
        return b
    

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)        
        self.masking = True
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, input):
        W = mm(self.weight, self.weight_mask, masking=self.masking)
        if self.bias is not None:
            b = mm(self.bias, self.bias_mask, masking=self.masking)
        else:
            b = self.bias
        return F.linear(input, W, b)
    
    def __repr__(self):
        return f'MaskedLinear(in_features={self.weight.size(1)}, out_features={self.weight.size(0)}, bias={self.bias is not None})'
    
    def __str__(self):
        return f'MaskedLinear(in_features={self.weight.size(1)}, out_features={self.weight.size(0)}, bias={self.bias is not None})'


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.masking = True
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        self.variance_epsilon = eps

    def forward(self, x):
        # layer norm should always be calculated in float32
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x / torch.sqrt(variance + self.variance_epsilon)

        if self.weight.dtype == torch.float16:
            x = x.to(torch.float16)
        W = mm(self.weight, self.weight_mask, masking=self.masking)
        return W * x
    
    def __repr__(self):
        return 'MaskedT5LayerNorm()'
    
    def __str__(self):
        return 'MaskedT5LayerNorm()'


class Embedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None, _freeze: bool = False,
                 device=None, dtype=None) -> None:
        super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight, _freeze, device, dtype)
        self.masking = True
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
    
    def forward(self, input: Tensor) -> Tensor:
        W = mm(self.weight, self.weight_mask, masking=self.masking)
        return F.embedding(
            input, W, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
    
    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        if self.masking is not False:
            s += ', masking=True'
        return s.format(**self.__dict__)
    
    def __repr__(self):
        return 'MaskedEmbedding'
    
    def __str__(self):
        return 'MaskedEmbedding'
    

def mask_pretrained_t5(model, device, dtype, skip_ln=False, skip_emb=False):

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear) and 'lm_head' not in name:
            state_dict = deepcopy(layer.state_dict())
            name = re.sub(r'(\.)(\d+)(\.)', r'[\2].', name)
            exec(f'model.{name} = Linear( \
                in_features=layer.in_features, \
                out_features=layer.out_features, \
                bias=layer.bias is not None \
            )')
            exec(f'model.{name}.load_state_dict(state_dict, False)')
        
        elif 'layer_norm' in name:
            if skip_ln: continue
            state_dict = deepcopy(layer.state_dict())
            name = re.sub(r'(\.)(\d+)(\.)', r'[\2].', name)
            exec(f'model.{name} = T5LayerNorm( \
                hidden_size=layer.weight.size(0) \
            )')
            exec(f'model.{name}.load_state_dict(state_dict, False)')

        elif isinstance(layer, nn.Embedding) and not 'embed_tokens' in name: # torch.nn.modules.sparse.Embedding
            if skip_emb: continue
            state_dict = deepcopy(layer.state_dict())
            name = re.sub(r'(\.)(\d+)(\.)', r'[\2].', name)
            exec(f'model.{name} = Embedding( \
                num_embeddings=layer.num_embeddings, \
                embedding_dim=layer.embedding_dim, \
                padding_idx=layer.padding_idx, \
                max_norm=layer.max_norm, \
                norm_type=layer.norm_type, \
                scale_grad_by_freq=layer.scale_grad_by_freq, \
                sparse=layer.sparse, \
                _weight=layer.weight, \
                _freeze=layer.weight.requires_grad, \
                device=device, \
                dtype=dtype \
            )')
            exec(f'model.{name}.load_state_dict(state_dict, False)')

            if 'shared' in name:
                model.transformer.encoder.embed_tokens = model.transformer.shared
                model.transformer.decoder.embed_tokens = model.transformer.shared
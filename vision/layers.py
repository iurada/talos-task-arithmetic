import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from typing import Optional, Tuple
from torch import Tensor
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from copy import deepcopy
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
        return 'MaskedLinear'
    
    def __str__(self):
        return 'MaskedLinear'


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, 
            dilation, groups, bias, padding_mode)
        self.masking = True
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        W = mm(self.weight, self.weight_mask, masking=self.masking)
        if self.bias is not None:
            b = mm(self.bias, self.bias_mask, masking=self.masking)
        else:
            b = self.bias
        return self._conv_forward(input, W, b)

    def __repr__(self):
        return 'MaskedConv2d'
    
    def __str__(self):
        return 'MaskedConv2d'


class NonDynamicallyQuantizableLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias)


class MultiheadAttention(nn.MultiheadAttention):
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn,
                 kdim, vdim, batch_first, device, dtype)
        self.masking = True
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)

            self.register_buffer('in_proj_weight_mask', None)
            self.register_buffer('q_proj_weight_mask', torch.ones(self.q_proj_weight.shape))
            self.register_buffer('k_proj_weight_mask', torch.ones(self.k_proj_weight.shape))
            self.register_buffer('v_proj_weight_mask', torch.ones(self.v_proj_weight.shape))
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

            self.register_buffer('in_proj_weight_mask', torch.ones(self.in_proj_weight.shape))
            self.register_buffer('q_proj_weight_mask', None)
            self.register_buffer('k_proj_weight_mask', None)
            self.register_buffer('v_proj_weight_mask', None)
        
        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
            self.register_buffer('in_proj_bias_mask', torch.ones(self.in_proj_bias.shape))
        else:
            self.register_parameter('in_proj_bias', None)
            self.register_buffer('in_proj_bias_mask', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))

            self.register_buffer('bias_k_mask', torch.ones(self.bias_k.shape))
            self.register_buffer('bias_v_mask', torch.ones(self.bias_v.shape))
        else:
            self.bias_k = self.bias_v = None

            self.register_buffer('bias_k_mask', None)
            self.register_buffer('bias_v_mask', None)

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super().__setstate__(state)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        
        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )


        why_not_fast_path = ''
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x is None or x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any([x is not None and x.requires_grad for x in tensor_args]):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    mm(self.in_proj_weight, self.in_proj_weight_mask, masking=self.masking),
                    mm(self.in_proj_bias, self.in_proj_bias_mask, masking=self.masking),
                    mm(self.out_proj.weight, self.out_proj.weight_mask, masking=self.masking),
                    mm(self.out_proj.bias, self.out_proj.bias_mask, masking=self.masking),
                    merged_mask,
                    need_weights,
                    average_attn_weights,
                    mask_type)

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                mm(self.in_proj_weight, self.in_proj_weight_mask, masking=self.masking), mm(self.in_proj_bias, self.in_proj_bias_mask, masking=self.masking),
                mm(self.bias_k, self.bias_k_mask, masking=self.masking), mm(self.bias_v, self.bias_v_mask, masking=self.masking), self.add_zero_attn,
                self.dropout, mm(self.out_proj.weight, self.out_proj.weight_mask, masking=self.masking), mm(self.out_proj.bias, self.out_proj.bias_mask, masking=self.masking),
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=mm(self.q_proj_weight, self.q_proj_weight_mask, masking=self.masking), k_proj_weight=mm(self.k_proj_weight, self.k_proj_weight_mask, masking=self.masking),
                v_proj_weight=mm(self.v_proj_weight, self.v_proj_weight_mask, masking=self.masking),
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                mm(self.in_proj_weight, self.in_proj_weight_mask, masking=self.masking), mm(self.in_proj_bias, self.in_proj_bias_mask, masking=self.masking),
                mm(self.bias_k, self.bias_k_mask, masking=self.masking), mm(self.bias_v, self.bias_v_mask, masking=self.masking), self.add_zero_attn,
                self.dropout, mm(self.out_proj.weight, self.out_proj.weight_mask, masking=self.masking), mm(self.out_proj.bias, self.out_proj.bias_mask, masking=self.masking),
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def merge_masks(self, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                    query: Tensor) -> Tuple[Optional[Tensor], Optional[int]]:
        r"""
        Determine mask type and combine masks if necessary. If only one mask is provided, that mask
        and the corresponding mask type will be returned. If both masks are provided, they will be both
        expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
        and mask type 2 will be returned
        Args:
            attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
            key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
            query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
        Returns:
            merged_mask: merged mask
            mask_type: merged mask type (0, 1, or 2)
        """
        mask_type: Optional[int] = None
        merged_mask: Optional[Tensor] = None

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask

        if attn_mask is not None:
            # In this branch query can't be a nested tensor, so it has a shape
            batch_size, seq_len, _ = query.shape
            mask_type = 2

            # Always expands attn_mask to 4D
            if attn_mask.dim() == 3:
                attn_mask_expanded = attn_mask.view(batch_size, -1, seq_len, seq_len)
            else:  # attn_mask.dim() == 2:
                attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(batch_size, self.num_heads, -1, -1)
            merged_mask = attn_mask_expanded

            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)
                merged_mask = attn_mask_expanded + key_padding_mask_expanded

        # no attn_mask and no key_padding_mask, returns None, None
        return merged_mask, mask_type
    
    def __repr__(self):
        return 'MaskedMultiheadAttention'
    
    def __str__(self):
        return 'MaskedMultiheadAttention'

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True,
                 bias: bool = True, device=None, dtype=None) -> None:
        super(LayerNorm, self).__init__(normalized_shape, eps, elementwise_affine, bias, device, dtype)
        self.masking = True
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, x: torch.Tensor):
        W = mm(self.weight, self.weight_mask, masking=self.masking)
        if self.bias is not None:
            b = mm(self.bias, self.bias_mask, masking=self.masking)
        else:
            b = self.bias
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, W, b, self.eps)
        return x.to(orig_type)
    
    def __repr__(self):
        return 'MaskedLayerNorm'
    
    def __str__(self):
        return 'MaskedLayerNorm'


def mask_pretrained_vit(model, device, dtype, skip_ln=False):

    for name, layer in model.named_modules():

        if isinstance(layer, nn.Conv2d):
            state_dict = deepcopy(layer.state_dict())
            name = re.sub(r'(\.)(\d+)(\.)', r'[\2].', name)
            exec(f'model.{name} = Conv2d( \
                in_channels=layer.in_channels, \
                out_channels=layer.out_channels, \
                kernel_size=layer.kernel_size, \
                stride=layer.stride, \
                padding=layer.padding, \
                dilation=layer.dilation, \
                groups=layer.groups, \
                bias = layer.bias is not None, \
                padding_mode=layer.padding_mode \
            )')
            exec(f'model.{name}.load_state_dict(state_dict, False)')
            
        elif isinstance(layer, nn.MultiheadAttention):
            state_dict = deepcopy(layer.state_dict())
            name = re.sub(r'(\.)(\d+)(\.)', r'[\2].', name)
            exec(f'model.{name} = MultiheadAttention( \
                embed_dim=layer.embed_dim, \
                num_heads=layer.num_heads, \
                dropout=layer.dropout, \
                bias=layer.in_proj_bias is not None, \
                add_bias_kv=layer.bias_k is not None, \
                add_zero_attn=layer.add_zero_attn, \
                kdim=layer.kdim, \
                vdim=layer.vdim, \
                batch_first=layer.batch_first, \
                device=device, \
                dtype=dtype \
            )')
            exec(f'model.{name}.load_state_dict(state_dict, False)')
        
        elif isinstance(layer, nn.Linear) and 'classification' not in name and 'attn' not in name:
            state_dict = deepcopy(layer.state_dict())
            name = re.sub(r'(\.)(\d+)(\.)', r'[\2].', name)
            exec(f'model.{name} = Linear( \
                in_features=layer.in_features, \
                out_features=layer.out_features, \
                bias=layer.bias is not None \
            )')
            exec(f'model.{name}.load_state_dict(state_dict, False)')
        
        elif isinstance(layer, nn.LayerNorm):
            if skip_ln: continue
            state_dict = deepcopy(layer.state_dict())
            name = re.sub(r'(\.)(\d+)(\.)', r'[\2].', name)
            exec(f'model.{name} = LayerNorm( \
                normalized_shape=layer.normalized_shape, \
                eps=layer.eps, \
                elementwise_affine=layer.elementwise_affine, \
                bias=layer.bias is not None, \
                device=device, \
                dtype=dtype \
            )')
            exec(f'model.{name}.load_state_dict(state_dict, False)')

            
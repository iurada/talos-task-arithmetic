import torch
import numpy as np
import gc
import torch.nn as nn
import torch.nn.functional as F
import layers
import copy
from tqdm import tqdm
from datasets.common import maybe_dictionarize
from heads import get_classification_head
from layers import Conv2d, MultiheadAttention, Linear
from task_vectors import NonLinearTaskVector
from modeling import ImageClassifier

def masks(module):
    r"""Returns an iterator over modules masks, yielding the mask.
    """
    for name, buf in module.named_buffers():
        if "mask" in name:
            yield buf

def trainable(module):
    r"""Returns boolean whether a module is trainable.
    """
    return not isinstance(module, (nn.Identity))

def prunable(module, batchnorm, residual):
    r"""Returns boolean whether a module is prunable.
    """
    isprunable = isinstance(module, (layers.MultiheadAttention, layers.Linear, layers.Conv2d, layers.LayerNorm))
    return isprunable

def parameters(model):
    r"""Returns an iterator over models trainable parameters, yielding just the
    parameter tensor.
    """
    for module in filter(lambda p: trainable(p), model.modules()):
        for param in module.parameters(recurse=False):
            yield param

def masked_parameters(model, bias=False, batchnorm=False, residual=False):
    r"""Returns an iterator over models prunable parameters, yielding both the
    mask and parameter tensors.
    """
    for module in filter(lambda p: prunable(p, batchnorm, residual), model.modules()):
        for mask, param in zip(masks(module), module.parameters(recurse=False)):
            #if param is not module.bias or bias is True:
            yield mask, param

class Pruner:
    def __init__(self, masked_parameters):
        self.masked_parameters = list(masked_parameters)
        self.scores = {}

    def score(self, model, loss, dataloader, device):
        raise NotImplementedError

    @torch.no_grad()
    def _global_copy(self, sparsity):
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        global_masks = torch.cat([torch.flatten(m).cpu() for m, p in self.masked_parameters if id(p) in self.scores])
        t, _ = torch.kthvalue(global_scores[global_masks == 1.0], min(int(sparsity * global_scores.numel()), global_scores[global_masks == 1.0].numel()-1))
        print('[+] Threshold:', t)
        tot, not_masked = 0, 0
        for mask, param in self.masked_parameters:
            if id(param) in self.scores:
                score = self.scores[id(param)]
                score[mask != 1.0] = torch.inf
                final_score = torch.ones_like(score)
                final_score.mul_(torch.where(score > t, 0.0, 1.0))
                mask.copy_(torch.where(score > t, 0.1, 1.0))
                tot += final_score.numel()
                not_masked += final_score.sum().item()
                setattr(param, 'score', final_score.clone().detach().cpu())
        print('[+] Remaining weights:', not_masked / tot, not_masked, tot)
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _random_copy(self, sparsity):
        for mask, param in self.masked_parameters:
            if id(param) in self.scores:
                score = torch.rand_like(self.scores[id(param)])
                dw_threshold, _ = torch.kthvalue(torch.flatten(score), int(sparsity * score.numel())) #!
                score.copy_(torch.where(score <= dw_threshold, 0.0, 1.0))
                setattr(param, 'score', score.clone().detach().cuda())

    @torch.no_grad()
    def _mag_copy(self, sparsity):
        global_scores = torch.cat([torch.flatten(v).abs() for _, v in self.masked_parameters if id(v) in self.scores])
        dw_threshold, _ = torch.kthvalue(global_scores, int(sparsity * global_scores.numel()))
        for mask, param in self.masked_parameters:
            if id(param) in self.scores:
                score = param.abs()
                dw_threshold, _ = torch.kthvalue(torch.flatten(score), int(sparsity * score.numel())) #!
                score.copy_(torch.where(score <= dw_threshold, 0.0, 1.0))
                setattr(param, 'score', score.clone().detach().cuda())
        
    @torch.no_grad()
    def _global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally.
        """
        # Threshold scores
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((1.0 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for mask, param in self.masked_parameters:
                if id(param) not in self.scores: continue
                score = self.scores[id(param)].to(mask.device)
                zero = torch.tensor([0.1]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))
                setattr(param, 'score', mask.clone().detach().cuda())
    
    @torch.no_grad()
    def _local_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level parameter-wise.
        """
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1.0 - sparsity) * score.numel())
            if not k < 1:
                threshold, _ = torch.kthvalue(torch.flatten(score), k)
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def mask(self, sparsity, scope, coeff=0.05, thresh=1e-6):
        r"""Updates masks of model with scores by sparsity according to scope.
        """
        if scope == 'global':
            self._global_mask(sparsity)
        if scope == 'global_copy':
            self._global_copy(sparsity, coeff)
        if scope == 'global_copyv2':
            self._global_copyv2(sparsity, coeff)
        if scope == 'global_copy_proplr':
            self._global_copy(sparsity, coeff)
        if scope == 'global_zero_thresh':
            self._global_zero_thresh(thresh)
        if scope == 'random_copy':
            self._random_copy(sparsity)
        if scope == 'mag_copy':
            self._mag_copy(sparsity)
        if scope == 'local':
            self._local_mask(sparsity)

    @torch.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters.
        """
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model.
        """
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    # Based on https://github.com/facebookresearch/open_lth/blob/master/utils/tensor_utils.py#L43
    def shuffle(self):
        for mask, param in self.masked_parameters:
            shape = mask.shape
            perm = torch.randperm(mask.nelement())
            mask = mask.reshape(-1)[perm].reshape(shape)

    def invert(self):
        for v in self.scores.values():
            v.div_(v**2)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters.
        """
        remaining_params, total_params = 0, 0 
        for mask, param in self.masked_parameters:
            if id(param) in self.scores:
                remaining_params += param.score.clone().detach().cpu().numpy().sum()
            total_params += mask.numel()
        return remaining_params, total_params

class TaLoS(Pruner):

    def __init__(self, masked_parameters, *args, **kwargs):
        super(TaLoS, self).__init__(masked_parameters)
        self.R = 1

    def score(self, model, loss, dataloader, device, batch_limit, *args, **kwargs):
        model = model.to(device)
        model.eval()
        
        for m, p in masked_parameters(model):
            m.requires_grad = False
            p.requires_grad = True
            self.scores[id(p)] = torch.zeros_like(p).cpu()

        for data in tqdm(dataloader):
            data = maybe_dictionarize(data)
            input = data['images'].to(device)
            
            for _ in range(self.R):
                logits = model(input)
                outdx = torch.distributions.Categorical(logits=logits).sample().unsqueeze(1).detach()
                samples = logits.gather(1, outdx)

                for idx in range(data.size(0)):
                    model.zero_grad()
                    torch.autograd.backward(samples[idx], retain_graph=True)
                    for m, p in masked_parameters(model):
                        if p.requires_grad and hasattr(p, 'grad') and p.grad is not None:
                            self.scores[id(p)] += torch.clone(p.grad.data.pow(2) * (m!=0).float()).detach().cpu()

        for m, p in masked_parameters(model):
            if p.requires_grad and hasattr(p, 'grad') and p.grad is not None:
                p.grad.data.zero_()
            m.requires_grad = False
            p.requires_grad = True


class LoTA(Pruner):

    def __init__(self, masked_parameters, *args, **kwargs):
        super(LoTA, self).__init__(masked_parameters)
        self.epochs = 1
    
    def score(self, model, dataset_name, args):
        zs_ckpt = f"{args.save}/{dataset_name}/zeroshot.pt"
        ft_ckpt = f"{args.save}/{dataset_name}/finetuned.pt"
        image_encoder = NonLinearTaskVector(zs_ckpt, ft_ckpt).apply_to(zs_ckpt, scaling_coef=1.0)
        classification_head = get_classification_head(args, dataset_name)
        ft_model = ImageClassifier(image_encoder, classification_head)

        layers.mask_pretrained_vit(ft_model, args.device, torch.float32, skip_ln=False)
        ft_model = ft_model.to(args.device)

        with torch.no_grad():
            for (mf, pf), (mp, pp) in zip(masked_parameters(ft_model), self.masked_parameters):
                self.scores[id(pp)] = torch.clone(pf - pp).detach().abs_().cuda()

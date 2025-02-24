import abc
import os

import torch
import torch.nn as nn
from functorch import jvp, make_functional_with_buffers

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from T5Wrapper import T5Wrapper
from utils import DotDict
import re


class LinearizedModule(nn.Module):
    """Creates a linearized version of a nn.Module.

    The linearized version of a model is a proper PyTorch model and can be
    trained as any other nn.Module.

    Args:
        model (nn.Module): The model to linearize. The trainable parameters of
            the linearized model will be initialized to the parameters of this
            model.
        init_model (nn.Module): A model of the same type as `model` containing
            the parameters around which the model is initialized. If not
            provided, `model` is used as the initialization model.
    """

    def __init__(self, model: nn.Module, init_model: nn.Module = None) -> None:
        """Initializes the linearized model."""
        super().__init__()
        if init_model is None:
            init_model = model

        func0, params0, self.buffers0 = make_functional_with_buffers(
            init_model.eval(), disable_autograd_tracking=True
        )
        self.func0 = lambda params, x: func0(params, self.buffers0, x)

        _, params, _ = make_functional_with_buffers(
            model, disable_autograd_tracking=True
        )

        self.params = nn.ParameterList(params)
        self.params0 = nn.ParameterList(params0)
        self.weight = self.params
        
        self._model_name = model.__class__.__name__

        # The intial parameters are not trainable.
        for p in self.params0:
            p.requires_grad = False

        # The params are.
        for p in self.params:
            p.requires_grad = True

    def __call__(self, x) -> torch.Tensor:
        """Computes the linearized model output using a first-order Taylor decomposition."""
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        out, dp = jvp(
            lambda param: self.func0(param, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return out + dp


class LinearizedT5Wrapper(abc.ABC, nn.Module):
    """Creates a linearized version of an image encoder."""

    def __init__(
        self, args=None, keep_lang=False, transformer=None, init_transformer=None, tokenizer=None
    ):
        super().__init__()
        if transformer is None:
            pretrainedModel_name = f'google-t5/{args.model}'
            transformer = AutoModelForSeq2SeqLM.from_pretrained(pretrainedModel_name)

        if init_transformer is None:
            init_transformer = transformer

        if tokenizer is None:
            pretrainedModel_name = f'google-t5/{args.model}'
            tokenizer = AutoTokenizer.from_pretrained(pretrainedModel_name, model_max_length=args.max_seq_len)
        
        transformer.module = LinearizedModule(transformer.module, init_transformer.module)

        self._model_name = self._get_name(args.model)
        self.model = T5Wrapper(transformer, tokenizer)
        self.tokenizer = self.model.tokenizer

    def _get_name(self, model_name):
        if "__pretrained__" in model_name:
            model_name, _ = model_name.split("__pretrained__", "")
        return model_name

    def forward(self, batch):
        return self.model.forward(batch)

    def compute_logProb_ofAllChoices(
        self,
        input_ids,
        input_masks,
        allChoices_ids,
        allChoices_masks,
        length_normalization
    ):
        return self.model.compute_logProb_ofAllChoices(
            input_ids,
            input_masks,
            allChoices_ids,
            allChoices_masks,
            length_normalization)

    def __call__(self, x):
        return self.forward(x)
    
    def predict_mulChoice(self, batch, length_normalization):
        return self.model.predict_mulChoice(batch, length_normalization)

    def save(self, filename):
        """Saves the linearized image encoder.

        We save the model name in the state dict so that we can load the
        correct model when loading the linearized image encoder. Directly using
        torch.save would not work becuse func0 is not serializable.

        Args:
            filename (str): The path to save the taylorized image encoder.
        """
        if os.path.dirname(filename) != "":
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        state_dict = self.state_dict()
        state_dict["model_name"] = self._model_name

        torch.save(state_dict, filename)

    @classmethod
    def load(cls, filename):
        """Loads a linearized image encoder.

        It first loads the state dict with the model name and then creates the
        correct model and loads the state dict.

        Args:
            filename (str): The path to the taylorized image encoder.

        Returns:
            LinearizedImageEncoder: The loaded taylorized image encoder.
        """
        print(f"Loading transformer from {filename}")
        state_dict = torch.load(filename, map_location="cpu")

        # ImageEncoder expects a DotDict
        args = DotDict({"model": state_dict["model_name"]})
        taylorized_encoder = cls(args)

        # Remove the model name from the state dict so that we can load the
        # model.
        state_dict.pop("model_name")
        taylorized_encoder.load_state_dict(state_dict, strict=False)
        return taylorized_encoder
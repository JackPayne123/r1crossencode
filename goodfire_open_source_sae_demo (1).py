# -*- coding: utf-8 -*-
"""Goodfire -- Open Source SAE Demo.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1IBMQtJqy8JiRk1Q48jDEgTISmtxhlCRL

# Goodfire Llama series SAEs

Before getting started making sure you've added your HF_TOKEN and GOODFIRE_API_KEY to your Colab secrets and granted this notebook access.

Learn more here: https://www.goodfire.ai/blog/sae-open-source-announcement/

## Install nnsight, huggingface_hub, and the Goodfire SDK

nnsight is a package for mechanistic interpretability work by our good friends at NDIF: https://nnsight.net
"""

!pip install nnsight==0.3.7

"""Use huggingface_hub to download the SAE"""

!pip install huggingface_hub

"""Use the Goodfire SDK to search features."""

!pip install goodfire

"""## Import dependencies"""

import torch
from typing import Optional, Callable

import nnsight
from nnsight.intervention import InterventionProxy

"""## Specify which language model, which SAE to use, and which layer"""

MODEL_NAME = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
SAE_NAME = 'Llama-3.1-8B-Instruct-SAE-l19'
SAE_LAYER = 'model.layers.19'
EXPANSION_FACTOR = 16 if SAE_NAME == 'Llama-3.1-8B-Instruct-SAE-l19' else 8

"""## Define SAE class"""

class SparseAutoEncoder(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.device = device
        self.encoder_linear = torch.nn.Linear(d_in, d_hidden)
        self.decoder_linear = torch.nn.Linear(d_hidden, d_in)
        self.dtype = dtype
        self.to(self.device, self.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of data using a linear, followed by a ReLU."""
        return torch.nn.functional.relu(self.encoder_linear(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a batch of data using a linear."""
        return self.decoder_linear(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """SAE forward pass. Returns the reconstruction and the encoded features."""
        f = self.encode(x)
        return self.decode(f), f


def load_sae(
    path: str,
    d_model: int,
    expansion_factor: int,
    device: torch.device = torch.device("cpu"),
):
    sae = SparseAutoEncoder(
        d_model,
        d_model * expansion_factor,
        device,
    )
    sae_dict = torch.load(
        path, weights_only=True, map_location=device
    )
    sae.load_state_dict(sae_dict)

    return sae

"""## Define language model wrapper"""

InterventionInterface = Callable[[InterventionProxy], InterventionProxy]


class ObservableLanguageModel:
    def __init__(
        self,
        model: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.dtype = dtype
        self.device = device
        self._original_model = model

        self._model = nnsight.LanguageModel(
            self._original_model,
            device_map=device,
            torch_dtype=getattr(torch, dtype) if isinstance(dtype, str) else dtype
        )

        # Quickly run a trace to force model to download due to nnsight lazy download
        input_tokens = self._model.tokenizer.apply_chat_template([{"role": "user", "content": "hello"}])
        with self._model.trace(input_tokens):
          pass

        self.tokenizer = self._model.tokenizer

        self.d_model = self._attempt_to_infer_hidden_layer_dimensions()

        self.safe_mode = False  # Nnsight validation is disabled by default, slows down inference a lot. Turn on to debug.

    def _attempt_to_infer_hidden_layer_dimensions(self):
        config = self._model.config
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)

        raise Exception(
            "Could not infer hidden number of layer dimensions from model config"
        )

    def _find_module(self, hook_point: str):
        submodules = hook_point.split(".")
        module = self._model
        while submodules:
            module = getattr(module, submodules.pop(0))
        return module

    def forward(
        self,
        inputs: torch.Tensor,
        cache_activations_at: Optional[list[str]] = None,
        interventions: Optional[dict[str, InterventionInterface]] = None,
        use_cache: bool = True,
        past_key_values: Optional[tuple[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor], dict[str, torch.Tensor]]:
        cache: dict[str, torch.Tensor] = {}
        with self._model.trace(
            inputs,
            scan=self.safe_mode,
            validate=self.safe_mode,
            use_cache=use_cache,
            past_key_values=past_key_values,
        ):
            # If we input an intervention
            if interventions:
                for hook_site in interventions.keys():
                    if interventions[hook_site] is None:
                        continue

                    module = self._find_module(hook_site)

                    intervened_acts = interventions[
                        hook_site
                    ](module.output[0])
                    # We only modify module.output[0]
                    if use_cache:
                        module.output = (
                            intervened_acts,
                            module.output[1],
                        )
                    else:
                        module.output = (intervened_acts,)

            if cache_activations_at is not None:
                for hook_point in cache_activations_at:
                    module = self._find_module(hook_point)
                    cache[hook_point] = module.output.save()

            if not past_key_values:
                logits = self._model.output[0][:, -1, :].save()
            else:
                logits = self._model.output[0].squeeze(1).save()

            kv_cache = self._model.output.past_key_values.save()

        return (
            logits.value.detach(),
            kv_cache.value,
            {k: v[0].detach() for k, v in cache.items()},
        )

"""## Download and instantiate the Llama model

**This will take a while to download Llama from HuggingFace.**
"""

model = ObservableLanguageModel(
    MODEL_NAME,
)

"""Let's read some activations out from the model."""

input_tokens = model.tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "Hello, how are you?"},
    ],
    add_generation_prompt=True,
    return_tensors="pt",
)
logits, kv_cache, feature_cache = model.forward(
    input_tokens,
    cache_activations_at=[SAE_LAYER],
)

print(feature_cache[SAE_LAYER].shape)

"""## Download and instantiate the SAE

Download from HuggingFace
"""

from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id=f"Goodfire/{SAE_NAME}",
    filename=f"{SAE_NAME}.pth",
    repo_type="model"
)

file_path

"""Load the SAE"""

sae = load_sae(
    file_path,
    d_model=model.d_model,
    expansion_factor=EXPANSION_FACTOR,
    device=model.device,
)

"""You can use the SAE on its own"""

features = sae.encode(feature_cache[SAE_LAYER])
features.shape

"""## Use the Goodfire API to search for features"""

import goodfire
from google.colab import userdata

client = goodfire.Client(userdata.get('GOODFIRE_API_KEY'))

pirate_features = client.features.search('pirate', MODEL_NAME)
pirate_features

"""## Intervene on the model to change it's outputs"""

pirate_feature_index = pirate_features[0].index_in_sae
pirate_feature_index

def example_intervention(activations: InterventionProxy):
    features = sae.encode(activations).detach()
    reconstructed_acts = sae.decode(features).detach()
    error = activations - reconstructed_acts

    # Modify feature at index 0 across all batch positions and token positions
    features[:, :, [pirate_feature_index]] += 12

    # Very important to add the error term back in!
    return sae.decode(features) + error

input_tokens = model.tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "Hello, how are you?"},
    ],
    add_generation_prompt=True,
    return_tensors="pt",
)

for i in range(10):
  logits, kv_cache, feature_cache = model.forward(
      input_tokens,
      interventions={SAE_LAYER: example_intervention},
  )

  new_token = logits[-1].argmax(-1)
  input_tokens = torch.cat([input_tokens[0], new_token.unsqueeze(0).cpu()]).unsqueeze(0)

  print(model.tokenizer.decode(new_token), end="")
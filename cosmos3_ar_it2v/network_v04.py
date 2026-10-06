"""V0.4 packed-stream adapters around the native Cosmos network operations."""
from __future__ import annotations

import torch

from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork


class Cosmos3VFMNetworkV04(Cosmos3VFMNetwork):
    """Reuse timestep work and project only P, retaining the full H/P transformer."""

    def _embed_packed_timesteps(self, timesteps, packed_seq, target_dtype):
        if getattr(packed_seq, "it2v_v04_prediction_vision", None) is None:
            return super()._embed_packed_timesteps(timesteps, packed_seq, target_dtype)
        # H shares a timestep over each latent's spatial patches; P additionally
        # shares it over each chunk. Reuse the official embedder without changing
        # the sampled values, parameters, or RNG. In particular, expand in FP32
        # BEFORE casting: the backward reduction of repeated rows must not happen
        # in BF16, even though the transformer consumes BF16 embeddings.
        unique_timesteps, inverse = torch.unique(timesteps, return_inverse=True)
        with torch.autocast("cuda", enabled=True, dtype=torch.float32):
            embeds = self.time_embedder(unique_timesteps)
        return embeds.index_select(0, inverse).to(target_dtype)

    def _decode_vision(self, packed_seq, last_hidden_state, output_dict,
                       original_latent_shapes=None):
        prediction = getattr(packed_seq, "it2v_v04_prediction_vision", None)
        if prediction is None:
            return super()._decode_vision(packed_seq, last_hidden_state, output_dict,
                                          original_latent_shapes)
        # Only the output head uses this P-only view. Encoding, timestep addition,
        # positions and transformer attention still use the intact H/P pack, so
        # target losses retain their indirect gradients through history hidden KV.
        # Source tensors carry the original spatial dimensions before patch padding.
        output_dict["preds_vision"] = self._decode_grid_stream(
            prediction, last_hidden_state,
            vae2llm=self.vae2llm, llm2vae=self.llm2vae,
            latent_channel=self.latent_channel, patch_latent_dim=self.patch_latent_dim,
            original_latent_shapes=[tuple(token.shape[2:]) for token in prediction.tokens],
        )


def install_v04_network_adapters(net):
    """Specialize only this instance at the official pre-parallelization hook.

    Cosmos constructs its network class directly. Switching an already-created
    instance to this parameter-free subclass avoids copying its build lifecycle
    or patching a global constructor. Existing modules, state-dict keys and RNG
    are untouched; FSDP/compilation see the final class before they wrap it.
    """
    if type(net) is Cosmos3VFMNetworkV04:
        return
    if type(net) is not Cosmos3VFMNetwork:
        raise TypeError("V0.4 adapters require the native network before parallelization")
    net.__class__ = Cosmos3VFMNetworkV04

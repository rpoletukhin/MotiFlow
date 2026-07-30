import functools as fn

import torch
from torch import nn
from torch.nn import functional as F

from motiflow.models.components import ipa_pytorch
from motiflow.models.components.utils import (
    ConditioningEncoder,
    GaussianSmearing,
    get_fragment_composition_tensor,
    get_timestep_embedding,
)
from motiflow.utils.rigid_utils import Rigid


class Embedder(nn.Module):
    def __init__(self, model_conf, fragment_library):
        super(Embedder, self).__init__()
        torch.set_default_dtype(torch.float32)
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed
        self.use_sc = model_conf.use_self_conditioning

        self.cond_encoder = ConditioningEncoder(model_conf.conditioning, out_dim=model_conf.node_embed_size)

        # Dimensions
        vocab_size = self._embed_conf.vocab_size
        index_embed_size = self._embed_conf.index_embed_size
        t_embed_size = index_embed_size

        self.frag_embedder = nn.Embedding(vocab_size + 1, index_embed_size)

        node_embed_dims = t_embed_size + index_embed_size
        if self.use_sc:
            self.sc_projector = nn.Linear(self._embed_conf.vocab_size, self._embed_conf.index_embed_size)
            node_embed_dims += self._embed_conf.index_embed_size

        # Reserve space for appending the condition
        if model_conf.conditioning.type != "none":
            node_embed_dims += model_conf.node_embed_size

        # Node MLP
        node_embed_size = self._model_conf.node_embed_size
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        # Edge Geometry
        self.rbf_stop = self._model_conf.rbf_stop
        self.dist_encoder = GaussianSmearing(start=0.0, stop=self.rbf_stop, num_gaussians=self._embed_conf.rbf_dim)

        edge_in = (node_embed_size * 2) + self._embed_conf.rbf_dim
        edge_embed_size = self._model_conf.edge_embed_size
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )

        self.timestep_embedder = fn.partial(get_timestep_embedding, embedding_dim=self._embed_conf.index_embed_size)

        # Explicit fragment composition embeddings (for conditioning tasks)
        self.frag_comp_projection = None
        if model_conf.conditioning.type != "none":
            print("Initializing Explicit Fragment Composition Embeddings...")
            comp_tensor = get_fragment_composition_tensor(
                fragment_library, model_conf.embed.vocab_size, model_conf.conditioning.num_atom_types
            )
            self.register_buffer("frag_composition", comp_tensor)

            self.frag_comp_projection = nn.Linear(
                model_conf.conditioning.num_atom_types, model_conf.embed.index_embed_size
            )

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return (
            torch.cat(
                [
                    torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
                    torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
                ],
                dim=-1,
            )
            .float()
            .reshape([num_batch, num_res**2, -1])
        )

    def forward(self, cat_t, t, rigids_t, sc_logits, condition):
        B, N = cat_t.shape

        # Base Features
        frag_feats = self.frag_embedder(cat_t.long())

        # Explicit atomic information for fragments
        if self.frag_comp_projection is not None:
            current_counts = F.embedding(cat_t.long(), self.frag_composition)
            comp_embed = self.frag_comp_projection(current_counts)
            frag_feats = frag_feats + comp_embed

        t_embed = self.timestep_embedder(t)
        prot_t_embed = t_embed.unsqueeze(1).repeat(1, N, 1)

        input_list = [prot_t_embed, frag_feats]

        # Self Conditioning
        if self.use_sc:
            if sc_logits is None:
                sc_embed = torch.zeros((B, N, self._embed_conf.index_embed_size), device=cat_t.device)
            else:
                sc_probs = F.softmax(sc_logits, dim=-1)
                sc_embed = self.sc_projector(sc_probs)
            input_list.append(sc_embed)

        # Concatenate condition
        cond_embed = None
        if self.cond_encoder.cond_type != "none" and condition is not None:
            cond_embed = self.cond_encoder(condition)  # [B, C]
            cond_expanded = cond_embed.unsqueeze(1).repeat(1, N, 1)  # [B, N, C]
            input_list.append(cond_expanded)

        # Project Nodes
        node_raw = torch.cat(input_list, dim=-1)
        node_embed = self.node_embedder(node_raw)

        # Form Initial Abstract Edges
        pair_feats = self._cross_concat(node_embed, B, N)  # [B, N*N, 2*node_dim]

        # Geometric Features (Using Rigid Utils)
        # Load Rigid
        r = Rigid.from_tensor_7(rigids_t)

        # Distances (RBF)
        trans = r.get_trans()  # [B, N, 3]
        dists = torch.cdist(trans, trans)  # [B, N, N]
        rbf = self.dist_encoder(dists)  # [B, N, N, 16]

        # Concatenate and Project
        # Flatten geometry to [B, N*N, features]
        rbf_flat = rbf.reshape(B, N * N, -1)

        # Combined: Pair(Abstract) + RBF(Dist)
        edge_raw = torch.cat([pair_feats, rbf_flat], dim=-1)

        edge_embed = self.edge_embedder(edge_raw)  # [B, N*N, edge_dim]
        edge_embed = edge_embed.reshape(B, N, N, -1)

        return node_embed, edge_embed, cond_embed


class VectorFieldNetwork(nn.Module):
    def __init__(self, model_conf, flow_matcher, fragment_library):
        super(VectorFieldNetwork, self).__init__()
        self._model_conf = model_conf

        self.embedding_layer = Embedder(model_conf, fragment_library)
        self.flow_matcher = flow_matcher
        self.vectorfield = ipa_pytorch.IpaNetwork(model_conf, flow_matcher)

        if self._model_conf._do_fm_cat:
            V = int(self.flow_matcher._se3_conf.cat.vocab_size)
            in_dim = self._model_conf.node_embed_size
            # Expand the readout dimension.
            # We will concatenate the condition embedding directly to the node features
            if self._model_conf.conditioning.type != "none":
                # Add size of the condition embedding (which matches node_embed_size in our encoder)
                in_dim += self._model_conf.node_embed_size
            self.cat_head = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, V))

    def _run_backbone(
        self,
        cat_t,
        rigids_t,
        t,
        frag_mask,
        sc_logits=None,
        condition=None,
        spatial_t=None,
    ):
        """
        Helper to run Embedding + IPA for a specific state
        """
        # Frames as [batch, res, 7] tensors.
        bb_mask = frag_mask.type(torch.float32)  # [B, N]
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]

        # Initial embeddings of positional and relative indices.
        init_node_embed, init_edge_embed, cond_embed = self.embedding_layer(
            cat_t=cat_t, t=t, rigids_t=rigids_t, sc_logits=sc_logits, condition=condition
        )
        edge_embed = init_edge_embed * edge_mask[..., None]
        node_embed = init_node_embed * bb_mask[..., None]

        # Prepare inputs for IPA
        feats = {
            "frag_mask": frag_mask,
            "rigids_t": rigids_t,  # Expected in tensor_7 format
            "t": t if spatial_t is None else spatial_t,
        }

        # Run IPA
        return self.vectorfield(node_embed, edge_embed, feats, cond_embed=cond_embed), cond_embed

    def forward(self, input_feats):
        """
        Forward predicts data-directed fields and p(X^1 | X^t)
        for each item in the batch
        """
        cat_t = input_feats["cat_t"]
        condition = input_feats.get("condition", None)

        out, cond_embed = self._run_backbone(
            cat_t=cat_t,
            rigids_t=input_feats["rigids_t"],
            t=input_feats["t"],
            frag_mask=input_feats["frag_mask"],
            sc_logits=input_feats["sc_logits"],
            condition=condition,
            spatial_t=input_feats.get("t_se3"),
        )

        # SE(3) flow predictions
        pred_out = {
            "rot_vectorfield": out["rot_vectorfield"],
            "trans_vectorfield": out["trans_vectorfield"],
            "rigids": out["final_rigids"].to_tensor_7(),
        }

        if self._model_conf._do_fm_cat:
            node_embed = out["node_embed"]  # [B, N, C]

            # Direct injection
            if self._model_conf.conditioning.type != "none" and condition is not None:
                # Broadcast to all fragments: [B, 1, C] -> [B, N, C]
                N = node_embed.shape[1]
                cond_expanded = cond_embed.unsqueeze(1).repeat(1, N, 1)
                # Concatenate: [B, N, NodeDim] + [B, N, CondDim]
                logits_input = torch.cat([node_embed, cond_expanded], dim=-1)
                pred_out["cat_logits"] = self.cat_head(logits_input)
            else:
                pred_out["cat_logits"] = self.cat_head(node_embed)

        return pred_out

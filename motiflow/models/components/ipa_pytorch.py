import math
from typing import Optional

import torch
import torch.nn as nn

from motiflow.models.components.utils import (
    GaussianSmearing,
    Linear,
    TriangleInteractionBlock,
    flatten_final_dims,
    ipa_point_weights_init_,
    permute_final_dims,
)
from motiflow.utils.rigid_utils import Rigid


class TransitionBlock(nn.Module):
    def __init__(self, c):
        super(TransitionBlock, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(self.c)

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)
        s = s + s_initial
        s = self.ln(s)

        return s


class EdgeTransition(nn.Module):
    def __init__(
        self,
        node_embed_size,
        edge_embed_in,
        edge_embed_out,
        num_layers=2,
        node_dilation=2,
    ):
        super(EdgeTransition, self).__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(node_embed_size, bias_embed_size, init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed):
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat(
            [
                torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
                torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
            ],
            axis=-1,
        )
        edge_embed = torch.cat([edge_embed, edge_bias], axis=-1).reshape(batch_size * num_res**2, -1)
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(batch_size, num_res, num_res, -1)
        return edge_embed


class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        ipa_conf,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        super(InvariantPointAttention, self).__init__()
        self._ipa_conf = ipa_conf
        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z
        self.c_hidden = ipa_conf.c_hidden_per_head
        self.no_heads = ipa_conf.no_heads
        self.no_qk_points = ipa_conf.no_qk_points
        self.no_v_points = ipa_conf.no_v_points
        self.inf = inf
        self.eps = eps

        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((ipa_conf.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        # Project Q, K, V
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # Project Points (Local Frame)
        q_pts = self.linear_q_points(s)
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)  # Shape: [B, N, H*P, 3]

        kv_pts = self.linear_kv_points(s)
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)  # Shape: [B, N, H*P, 3]

        q_pts = r[..., None].apply(q_pts)
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))

        kv_pts = r[..., None].apply(kv_pts)
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))
        k_pts, v_pts = torch.split(kv_pts, [self.no_qk_points, self.no_v_points], dim=-2)

        # Attention Scores
        b = self.linear_b(z)

        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),
            permute_final_dims(k, (1, 2, 0)),
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement**2
        pt_att = sum(torch.unbind(pt_att, dim=-1))

        head_weights = self.softplus(self.head_weights).view(*((1,) * len(pt_att.shape[:-2]) + (-1, 1)))
        head_weights = head_weights * math.sqrt(1.0 / (3 * (self.no_qk_points * 9.0 / 2)))
        pt_att = pt_att * head_weights
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)

        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        # Values
        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)
        o = flatten_final_dims(o, 2)

        o_pt = torch.sum(
            (a[..., None, :, :, None] * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]),
            dim=-2,
        )
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        o_pt_dists = torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(o_pt_dists, 2)
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        pair_z = self.down_z(z).to(dtype=a.dtype)
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]
        s = self.linear_out(torch.cat(o_feats, dim=-1).to(dtype=z.dtype))

        return s


class RigidUpdate(nn.Module):
    """
    Split rigid update: rotation 3-dim (small linear), and a small MLP for translation.
    Produces a 6-dim vector [rot(3), trans(3)] for compose_q_update_vec.
    """

    def __init__(self, c_s, trans_hidden=256):
        super(RigidUpdate, self).__init__()
        self.c_s = c_s
        # Rotation head: small linear (init final zero)
        self.rot_linear = Linear(self.c_s, 3, init="final")
        # Translation head: small 2-layer MLP
        self.trans_mlp = nn.Sequential(
            Linear(self.c_s, trans_hidden, init="relu"),
            nn.ReLU(),
            Linear(trans_hidden, 3, init="final"),
        )
        self.trans_mlp[2].weight.data.mul_(0.01)
        self.trans_mlp[2].bias.data.fill_(0.0)

    def forward(self, s: torch.Tensor):
        rot = self.rot_linear(s)  # [B,N,3]
        trans = self.trans_mlp(s)  # [B,N,3]
        update = torch.cat([rot, trans], dim=-1)  # [B,N,6]
        return update


class IpaNetwork(nn.Module):
    def __init__(self, model_conf, flow_matcher):
        super(IpaNetwork, self).__init__()
        self._model_conf = model_conf
        ipa_conf = model_conf.ipa
        self._ipa_conf = ipa_conf
        self.flow_matcher = flow_matcher
        self.update_edge_all = ipa_conf.do_last_edge_update
        self.with_triangl_upd = model_conf.with_triangl_upd

        # Coordinate scaling helper
        self.scale_pos = lambda x: x * ipa_conf.coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)
        self.unscale_pos = lambda x: x / ipa_conf.coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)

        self.rbf_dim = self._model_conf.embed.rbf_dim
        self.rbf_stop = self._model_conf.embed.ipa_rbf_stop
        self.dist_encoder = GaussianSmearing(start=0.0, stop=self.rbf_stop, num_gaussians=self.rbf_dim)

        self.trunk = nn.ModuleDict()

        for b in range(ipa_conf.num_blocks):
            # Dynamic Edge Update Projector
            self.trunk[f"edge_update_proj_{b}"] = Linear(self.rbf_dim, model_conf.edge_embed_size, init="final")

            # Full Triangle Interaction
            if self.with_triangl_upd:
                self.trunk[f"triangle_block_{b}"] = TriangleInteractionBlock(
                    c_z=model_conf.edge_embed_size,
                    c_hidden=model_conf.edge_embed_size,
                )

            # Invariant Point Attention (Local Frame Geometry Mixing)
            self.trunk[f"ipa_{b}"] = InvariantPointAttention(ipa_conf)
            self.trunk[f"ipa_ln_{b}"] = nn.LayerNorm(ipa_conf.c_s)

            # Global Transformer (Set/Global mixing)
            self.trunk[f"skip_embed_{b}"] = Linear(
                self._model_conf.node_embed_size, self._ipa_conf.c_skip, init="final"
            )
            embed_dim = ipa_conf.c_s + self._ipa_conf.c_skip

            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=embed_dim,
                batch_first=True,
                dropout=0.0,
                norm_first=False,
            )
            self.trunk[f"global_transformer_{b}"] = torch.nn.TransformerEncoder(
                tfmr_layer, ipa_conf.seq_tfmr_num_layers
            )
            self.trunk[f"post_transformer_{b}"] = Linear(embed_dim, ipa_conf.c_s, init="final")

            # Transition Block (MLP)
            self.trunk[f"transition_{b}"] = TransitionBlock(c=ipa_conf.c_s)

            # Rigid Update (Predicts Flow) -- now split head
            self.trunk[f"rigid_update_{b}"] = RigidUpdate(ipa_conf.c_s, trans_hidden=ipa_conf.c_hidden_per_head)

            if b < ipa_conf.num_blocks - 1 or self.update_edge_all:
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f"edge_transition_{b}"] = EdgeTransition(
                    node_embed_size=ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

        # Setup Conditional Injectors
        self.cond_type = model_conf.conditioning.type
        if self.cond_type != "none":
            for b in range(ipa_conf.num_blocks):
                self.trunk[f"cond_inject_{b}"] = Linear(
                    model_conf.node_embed_size, model_conf.node_embed_size, init="default"
                )

    def forward(self, init_node_embed, edge_embed, input_feats, cond_embed=None):
        # Masks
        node_mask = input_feats["frag_mask"].type(torch.float32)
        edge_mask = node_mask[..., None] * node_mask[..., None, :]

        # Frames
        init_frames = input_feats["rigids_t"].type(torch.float32)
        curr_rigids = Rigid.from_tensor_7(torch.clone(init_frames))
        init_rigids = Rigid.from_tensor_7(init_frames)

        # Scale
        curr_rigids = self.scale_rigids(curr_rigids)
        node_embed = init_node_embed * node_mask[..., None]

        tri_mask = edge_mask

        for b in range(self._ipa_conf.num_blocks):
            # Dynamic Edge Update (Geometric Injection)
            trans = curr_rigids.get_trans()
            dists = torch.cdist(trans, trans)
            rbf_geo = self.dist_encoder(dists)
            geo_update = self.trunk[f"edge_update_proj_{b}"](rbf_geo)

            edge_embed = edge_embed + geo_update
            edge_embed = edge_embed * edge_mask.unsqueeze(-1)

            # Full Triangle Logic (Global Consistency)
            if self.with_triangl_upd:
                edge_embed = self.trunk[f"triangle_block_{b}"](edge_embed, mask=tri_mask)

            # IPA (Geometry)
            ipa_embed = self.trunk[f"ipa_{b}"](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask,
            )
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f"ipa_ln_{b}"](node_embed + ipa_embed)

            # Global Transformer (Set Mixing)
            global_in = torch.cat([node_embed, self.trunk[f"skip_embed_{b}"](init_node_embed)], dim=-1)
            global_out = self.trunk[f"global_transformer_{b}"](global_in, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f"post_transformer_{b}"](global_out)

            # Transition
            node_embed = self.trunk[f"transition_{b}"](node_embed)
            node_embed = node_embed * node_mask[..., None]

            # Inject Conditioning
            if self.cond_type != "none" and cond_embed is not None:
                # Project global vector to block-specific bias
                cond_bias = self.trunk[f"cond_inject_{b}"](cond_embed)  # [B, C]
                # Add to nodes
                node_embed = node_embed + cond_bias.unsqueeze(1) * node_mask[..., None]

            # Rigid Update
            rigid_update_vec = self.trunk[f"rigid_update_{b}"](node_embed)

            # Update the frames
            curr_rigids = curr_rigids.compose_q_update_vec(rigid_update_vec, node_mask[..., None])

            # Edge Update
            if b < self._ipa_conf.num_blocks - 1 or self.update_edge_all:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # Unscale
        curr_rigids = self.unscale_rigids(curr_rigids)
        t = input_feats["t"]

        # Calculate Flow (Vector Field)
        _, rot_vectorfield = self.flow_matcher.calc_rot_vectorfield(
            curr_rigids.get_rots().get_rot_mats(),
            init_rigids.get_rots().get_rot_mats(),
            t,
        )
        rot_vectorfield = rot_vectorfield * node_mask[..., None, None]

        trans_vectorfield = self.flow_matcher.calc_trans_vectorfield(
            curr_rigids.get_trans(),
            init_rigids.get_trans(),
            input_feats["t"][:, None, None],
        )
        trans_vectorfield = trans_vectorfield * node_mask[..., None]

        model_out = {
            "final_rigids": curr_rigids,
            "rot_vectorfield": rot_vectorfield,
            "trans_vectorfield": trans_vectorfield,
            "node_embed": node_embed,
            "edge_embed": edge_embed,
        }

        return model_out

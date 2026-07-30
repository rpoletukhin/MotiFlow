from collections import defaultdict

import numpy as np
import torch
from torch.utils import data

from motiflow.data.conditioning import validate_conditioning_type
from motiflow.data.loaders import ShardedDataSequence, load_processed_data
from motiflow.utils.rigid_helpers import assemble_rigid_mat


class FragmentDataset(data.Dataset):
    def __init__(self, split, data_conf, is_training):
        self._is_training = is_training
        self._data_conf = data_conf

        # Load on CPU so dataset construction is independent of the device used
        # When the artifacts were written.
        self.data_list = load_processed_data(data_conf.processed_data_dir, split)
        self.cond_type = validate_conditioning_type(str(data_conf.conditioning_type))
        self.library = torch.load(data_conf.library_path, map_location="cpu", weights_only=False)
        if not isinstance(self.library, dict) or not self.library:
            raise ValueError("Fragment library must be a non-empty dictionary.")

        library_ids = sorted(int(cid) for cid in self.library)
        expected_ids = list(range(len(library_ids)))
        if library_ids != expected_ids:
            raise ValueError(
                f"Fragment library IDs must be contiguous and zero-based; found {library_ids[:5]}...{library_ids[-5:]}."
            )

        configured_vocab_size = data_conf.get("vocab_size")
        self.vocab_size = len(self.library) if configured_vocab_size is None else int(configured_vocab_size)
        if self.vocab_size != len(self.library):
            raise ValueError(
                f"Configured vocab_size={self.vocab_size} but the library contains {len(self.library)} classes."
            )

        largest_symmetry_count = max(len(entry["symmetries"]) for entry in self.library.values())
        configured_s_max = data_conf.get("S_max")
        self.S_max = largest_symmetry_count if configured_s_max is None else int(configured_s_max)
        if self.S_max < largest_symmetry_count:
            raise ValueError(
                f"Configured S_max={self.S_max} is smaller than the library maximum {largest_symmetry_count}."
            )

        # Pre-process library symmetries into tensors
        self.lib_symmetries = {}
        for cid, entry in self.library.items():
            syms = torch.as_tensor(entry["symmetries"], dtype=torch.float32)  # (num_syms, 3, 3)
            num_syms = syms.shape[0]

            if num_syms > self.S_max:
                raise ValueError(f"Class {cid} has {num_syms} symmetries, > S_max {self.S_max}")

            # Pad rotational symmetries with identity matrices
            padded_syms = torch.eye(3).unsqueeze(0).repeat(self.S_max, 1, 1)  # (S_max, 3, 3)
            padded_syms[:num_syms] = syms

            # Mask (True for valid symmetries, False for padded)
            mask = torch.zeros(self.S_max, dtype=torch.bool)
            mask[:num_syms] = True

            self.lib_symmetries[cid] = (padded_syms, mask)

        # Add a unit matrix for the [MASK] state symmetry
        mask_mask = torch.zeros(self.S_max, dtype=torch.bool)
        mask_mask[0] = True
        self.lib_symmetries[self.vocab_size] = (
            torch.eye(3).unsqueeze(0).repeat(self.S_max, 1, 1),  # (S_max, 3, 3)
            mask_mask,
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if idx is None:
            return None

        item = self.data_list[idx]

        # Extract the raw data
        frag_ids = item["frag_ids"].long()
        trans_gt = item["trans"].float()
        rots_gt = item["rots"].float().clone()
        K = frag_ids.shape[0]

        frag_mask = torch.ones(K, dtype=torch.float32)

        # Center the fragments
        center_of_frags = trans_gt.mean(dim=0, keepdim=True)
        trans_gt_centered = trans_gt - center_of_frags

        # Build tensor for rotational symmetries for each fragment and mask
        mol_syms = torch.zeros((K, self.S_max, 3, 3), dtype=torch.float32)
        mol_sym_mask = torch.zeros((K, self.S_max), dtype=torch.bool)

        for k in range(K):
            cid = int(frag_ids[k].item())
            # Load symmetries
            if cid in self.lib_symmetries:
                s, m = self.lib_symmetries[cid]
            else:
                raise ValueError(f"Fragment ID {cid} not found in library.")
            # Fill Output Tensors
            mol_syms[k] = s
            mol_sym_mask[k] = m

            # # 2. Apply Random Frame Augmentation (Training Only) if Enabled
            if self._is_training and self._data_conf.use_frame_augmentation:
                num_valid = int(m.sum().item())

                if num_valid > 1:
                    # Sample index: 0 to num_valid-1
                    rand_idx = torch.randint(0, num_valid, (1,)).item()

                    # If rand_idx == 0 (Identity), rots_gt remains unchanged.
                    # If > 0, we apply the discrete symmetry.
                    if rand_idx > 0:
                        S = s[rand_idx]
                        # R_new = R_old @ S (Rotate the local frame)
                        rots_gt[k] = torch.matmul(rots_gt[k], S)

        # Assemble clean centered rigids from the (potentially augmented) rotations
        rigids_1_obj = assemble_rigid_mat(rots_gt, trans_gt_centered)
        # Convert rigids_1 to tensor_7 format
        rigids_1 = rigids_1_obj.to_tensor_7()

        cond_data = {}
        if self.cond_type == "composition":
            cond_data["condition"] = (
                item["composition_counts"].float() / 29
            )  # normalize atom counts by max atoms in QM9
        elif self.cond_type == "structure":
            cond_data["condition"] = item["fingerprint"].float()

        # Load Evaluation Target (Ground Truth)
        eval_target = torch.zeros(1)

        if self.cond_type == "composition":
            # Target is the integer counts (e.g., [6, 12, 0, 6, 0])
            eval_target = item["composition_counts"].float()
        elif self.cond_type == "structure":
            # Target is the fingerprint
            eval_target = item["fingerprint"].float()

        # Assemble Output
        out = {
            "frag_ids": frag_ids,
            "frag_mask": frag_mask,
            "rigids_1": rigids_1,
            "symmetries": mol_syms,
            "sym_mask": mol_sym_mask,
            "eval_target": eval_target,
            **cond_data,
        }

        return out


class TrainSampler(data.Sampler):
    def __init__(self, dataset, batch_size, include_partial_batches=True):
        """
        Groups dataset indices by sequence length and yields batches grouped by length.
        By default `include_partial_batches=True` so the leftover (partial) batch per length
        is included instead of being dropped.

        Args:
            dataset: dataset with `data_list` accessible and each item having 'frag_ids'.
            batch_size: desired batch_size used by DataLoader to collate mini-batches.
            include_partial_batches: if True, include the last smaller batch for each length group.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.epoch = 0
        self.include_partial_batches = include_partial_batches

        # Keep shard locality so one large split shard is loaded only once per epoch.
        self.shard_length_to_indices = []
        records = self.dataset.data_list
        if isinstance(records, ShardedDataSequence):
            global_offset = 0
            for shard_index in range(len(records.shard_paths)):
                length_to_indices = defaultdict(list)
                shard = records._load_shard(shard_index)
                for local_index, item in enumerate(shard):
                    length_to_indices[len(item["frag_ids"])].append(global_offset + local_index)
                self.shard_length_to_indices.append(length_to_indices)
                global_offset += len(shard)
        else:
            length_to_indices = defaultdict(list)
            for index, item in enumerate(records):
                length_to_indices[len(item["frag_ids"])].append(index)
            self.shard_length_to_indices.append(length_to_indices)

        self.num_samples = 0
        for length_to_indices in self.shard_length_to_indices:
            for indices in length_to_indices.values():
                if self.include_partial_batches:
                    self.num_samples += len(indices)
                else:
                    self.num_samples += (len(indices) // self.batch_size) * self.batch_size

    def __iter__(self):
        # Deterministic per-epoch RNG
        rng = np.random.default_rng(self.epoch)
        shard_order = np.arange(len(self.shard_length_to_indices))
        rng.shuffle(shard_order)
        final_indices = []
        for shard_index in shard_order:
            batches = []
            for indices in self.shard_length_to_indices[int(shard_index)].values():
                indices = np.array(indices)
                rng.shuffle(indices)
                num_full = len(indices) // self.batch_size
                for batch_index in range(num_full):
                    start = batch_index * self.batch_size
                    batches.append(indices[start : start + self.batch_size])
                if self.include_partial_batches and len(indices) % self.batch_size:
                    batches.append(indices[num_full * self.batch_size :])
            rng.shuffle(batches)
            final_indices.extend(int(index) for batch in batches for index in batch)
        return iter(final_indices)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return int(self.num_samples)

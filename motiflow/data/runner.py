import hashlib
import io
import itertools
import json
import os
import pickle
import time
from collections import deque
from multiprocessing import Pool, TimeoutError

import ase.io
import torch
from rdkit import Chem
from tqdm import tqdm

from motiflow.data.aligner import TopologyAligner
from motiflow.provenance import write_json_atomic


def read_xyz_file(file_handle, center: bool = True):
    """Exactly replicates original QM9 coordinate parsing."""
    content = [line.decode("UTF-8") for line in file_handle.readlines()]
    num_atoms = int(content[0])
    xyz_block = "".join(content[0 : num_atoms + 2]).replace("*^", "e")

    with io.StringIO(xyz_block) as f:
        atoms = ase.io.read(f, format="xyz")

    if center:
        atoms.positions -= atoms.positions.mean(axis=0)

    return atoms, content[-2].split()[0]


def extract_and_align(task):
    if rdBase := getattr(Chem, "rdBase", None):
        try:
            rdBase.DisableMultithreading()
        except (AttributeError, RuntimeError):
            pass

    try:
        if task.get("load_error"):
            return task["source_order"], task["sample_id"], "ERROR", task["load_error"]
        if task["dataset"] == "geom":
            src_raw = Chem.Mol(task["mol_bytes"])
            z = torch.tensor([a.GetAtomicNum() for a in src_raw.GetAtoms()], dtype=torch.long)
            pos = task["pos"]
        elif task["dataset"] == "qmugs":
            suppl = Chem.SDMolSupplier(task["sdf_path"], removeHs=False)
            src_raw = suppl[0] if len(suppl) > 0 else None
            if src_raw is None:
                return task["source_order"], task["sample_id"], "ERROR", "SDF Load Failed"
            conf = src_raw.GetConformer()
            pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
            z = torch.tensor([a.GetAtomicNum() for a in src_raw.GetAtoms()], dtype=torch.long)
    except Exception as e:
        return task["source_order"], task["sample_id"], "ERROR", f"Extraction failed: {str(e)}"

    aligner = TopologyAligner(**task["aligner_kwargs"])
    status, data = aligner.align(
        task["sample_id"],
        z,
        pos,
        task["smiles"],
        task["aux"],
        source_mol=src_raw,
    )
    if status == "OK":
        data.source_order = int(task["source_order"])
    return task["source_order"], task["sample_id"], status, data


def _run_worker_serialized(worker, task):
    """Return a byte payload so multiprocessing never shares Torch storage by file descriptor."""
    output = worker(task)
    return pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL)


def _terminate_pool_safely(pool):
    """Terminate a pool even if Python's result-handler thread has already failed."""
    try:
        pool.terminate()
        pool.join()
    except AssertionError:
        # multiprocessing.Pool.terminate() asserts when its result handler has
        # died with cached AsyncResults. Kill and reap the workers directly so
        # the original infrastructure failure is not hidden by cleanup.
        for process in getattr(pool, "_pool", ()):
            if process.is_alive():
                process.terminate()
        for process in getattr(pool, "_pool", ()):
            process.join()


def _run_tasks_supervised(tasks, num_workers, timeout_seconds, worker=None):
    """Yield worker results while hard-killing a pool that contains a stuck task.

    Only a rolling window of tasks is submitted. If one exceeds its wall-clock
    deadline, the pool is terminated, the timed-out task is reported, and other
    in-flight tasks are safely resubmitted to a fresh pool.
    """
    worker = worker or extract_and_align
    if num_workers < 1:
        for task in tasks:
            try:
                yield worker(task)
            except Exception as exc:
                yield task["source_order"], task["sample_id"], "ERROR", f"Worker crashed: {exc}"
        return

    timeout_seconds = None if timeout_seconds is None else float(timeout_seconds)
    task_iterator = iter(tasks)
    pending = deque()
    pool = Pool(processes=num_workers)
    pool_is_running = True

    def submit_window():
        while len(pending) < num_workers:
            try:
                task = next(task_iterator)
            except StopIteration:
                break
            pending.append((task, pool.apply_async(_run_worker_serialized, (worker, task)), time.monotonic()))

    try:
        submit_window()
        while pending:
            task, result, submitted_at = pending.popleft()
            try:
                if timeout_seconds is None or timeout_seconds <= 0:
                    payload = result.get()
                else:
                    remaining = max(0.0, timeout_seconds - (time.monotonic() - submitted_at))
                    payload = result.get(timeout=remaining)
                output = pickle.loads(payload)
            except TimeoutError:
                retry_tasks = [entry[0] for entry in pending]
                pending.clear()
                _terminate_pool_safely(pool)
                pool_is_running = False
                yield (
                    task["source_order"],
                    task["sample_id"],
                    "TIMEOUT",
                    f"Alignment exceeded {timeout_seconds:g} seconds",
                )
                task_iterator = iter(itertools.chain(retry_tasks, task_iterator))
                pool = Pool(processes=num_workers)
                pool_is_running = True
            except Exception as exc:
                output = (
                    task["source_order"],
                    task["sample_id"],
                    "ERROR",
                    f"Worker crashed: {exc}",
                )
                yield output
            else:
                yield output
            submit_window()
    finally:
        if pool_is_running:
            pool.close()
            pool.join()


def _shard_fingerprint(dataset, aligner_kwargs, shard_tasks):
    payload = {
        "dataset": dataset,
        "aligner_kwargs": aligner_kwargs,
        "sample_ids": [str(task["sample_id"]) for task in shard_tasks],
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_pipeline(
    tasks,
    dataset,
    output_path,
    temp_dir,
    batch_size,
    num_workers,
    aligner_kwargs,
    task_timeout_seconds=120.0,
    shard_group=None,
):
    for source_order, t in enumerate(tasks):
        t["source_order"] = source_order
        t["aligner_kwargs"] = aligner_kwargs

    failures = []

    if batch_size is None:
        if dataset == "qm9":
            import tarfile

            aligner = TopologyAligner(**aligner_kwargs)
            tf_path = tasks[0]["tar_path"]
            buffer = []

            with tarfile.open(tf_path, "r") as tf:
                members = sorted(tf.getmembers(), key=lambda m: m.name)
                for task in tqdm(tasks, desc="QM9 Process"):
                    try:
                        f = tf.extractfile(members[task["tar_index"]])
                        if not f:
                            continue

                        atoms, smiles = read_xyz_file(f, center=True)
                        z = torch.tensor(atoms.numbers, dtype=torch.long)
                        pos = torch.tensor(atoms.positions, dtype=torch.float)

                        status, data = aligner.align(task["sample_id"], z, pos, smiles, {})
                        if status == "OK":
                            data.source_order = int(task["source_order"])
                            buffer.append(data)
                        else:
                            failures.append(
                                {"sample_id": str(task["sample_id"]), "status": status, "message": str(data)}
                            )
                    except Exception as e:
                        failures.append({"sample_id": str(task["sample_id"]), "status": "ERROR", "message": str(e)})

            buffer.sort(key=lambda data: int(data.source_order))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(buffer, output_path)
            print(f"Saved {len(buffer)} molecules to {output_path}")
            return {
                "counts": {"requested": len(tasks), "resumed": 0, "succeeded": len(buffer), "failed": len(failures)},
                "failures": failures,
                "output_files": [output_path],
            }

    if batch_size is None or batch_size < 1:
        raise ValueError("Parallel alignment requires a positive batch_size.")

    split_name = os.path.basename(output_path).removesuffix("_aligned.pt")
    shard_dir = os.path.join(os.path.dirname(output_path), split_name)
    if shard_group:
        shard_dir = os.path.join(shard_dir, str(shard_group))
    os.makedirs(shard_dir, exist_ok=True)

    pending_shards = {}
    output_files = []
    resumed = 0
    succeeded = 0

    for shard_index, start in enumerate(range(0, len(tasks), batch_size)):
        shard_tasks = tasks[start : start + batch_size]
        shard_path = os.path.join(shard_dir, f"shard_{shard_index:05d}.pt")
        manifest_path = os.path.join(shard_dir, f"shard_{shard_index:05d}.manifest.json")
        fingerprint = _shard_fingerprint(dataset, aligner_kwargs, shard_tasks)

        if os.path.exists(shard_path) or os.path.exists(manifest_path):
            if not (os.path.exists(shard_path) and os.path.exists(manifest_path)):
                raise RuntimeError(f"Incomplete alignment shard pair: {shard_path}")
            with open(manifest_path) as handle:
                shard_manifest = json.load(handle)
            if shard_manifest.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    f"Alignment shard fingerprint mismatch at {shard_path}. "
                    "Use a clean output directory for changed inputs or alignment settings."
                )
            records = torch.load(shard_path, map_location="cpu")
            resumed += len(shard_tasks)
            succeeded += len(records)
            failures.extend(shard_manifest.get("failures", []))
            output_files.extend([shard_path, manifest_path])
        else:
            pending_shards[shard_index] = {
                "tasks": shard_tasks,
                "path": shard_path,
                "manifest_path": manifest_path,
                "fingerprint": fingerprint,
                "records": [],
                "failures": [],
                "remaining": len(shard_tasks),
            }

    work_tasks = itertools.chain.from_iterable(entry["tasks"] for entry in pending_shards.values())
    work_count = sum(entry["remaining"] for entry in pending_shards.values())
    iterator = (
        _run_tasks_supervised(
            work_tasks,
            num_workers=num_workers,
            timeout_seconds=task_timeout_seconds,
        )
        if work_count
        else ()
    )
    for source_order, sample_id, status, data in tqdm(iterator, total=work_count, desc="Aligning"):
        shard_index = int(source_order) // batch_size
        entry = pending_shards[shard_index]
        if status == "OK":
            entry["records"].append(data)
            succeeded += 1
        else:
            failure = {"sample_id": str(sample_id), "status": status, "message": str(data)}
            entry["failures"].append(failure)
            failures.append(failure)
        entry["remaining"] -= 1

        if entry["remaining"] == 0:
            entry["records"].sort(key=lambda record: int(record.source_order))
            torch.save(entry["records"], entry["path"])
            shard_manifest = {
                "fingerprint": entry["fingerprint"],
                "dataset": dataset,
                "aligner_kwargs": aligner_kwargs,
                "requested_ids": [str(task["sample_id"]) for task in entry["tasks"]],
                "succeeded_ids": [str(record.sample_id) for record in entry["records"]],
                "failures": entry["failures"],
            }
            write_json_atomic(shard_manifest, entry["manifest_path"])
            output_files.extend([entry["path"], entry["manifest_path"]])

    output_files.sort()
    print(f"Saved {succeeded} molecules in deterministic shards under {shard_dir}")
    return {
        "counts": {
            "requested": len(tasks),
            "resumed": resumed,
            "succeeded": succeeded,
            "failed": len(failures),
        },
        "failures": failures,
        "output_files": output_files,
    }

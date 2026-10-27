"""Make lossless, independently downloadable HDF5 episodes from a consolidated dataset."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import tempfile

import h5py
import numpy as np


REQUIRED_FIELDS = {
    "actions", "states", "camera_0", "camera_1", "camera_2",
    "cam_0_patch_embd", "cam_1_patch_embd", "cam_tactile_patch_embd",
}
OPTIONAL_FIELDS = {
    "cam_0_cls_embd", "cam_1_cls_embd", "cam_tactile_cls_embd",
    "ee_states", "gripper_states", "joint_states",
}
TERMINAL_OBSERVATION_FIELDS = {
    "camera_0", "camera_1", "camera_2", "ee_states", "gripper_states", "joint_states",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def source_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev, "inode": stat.st_ino,
    }


def block_length(dataset):
    frame_bytes = int(np.prod(dataset.shape[1:])) * dataset.dtype.itemsize
    return max(1, min(32, (4 * 1024 * 1024) // max(1, frame_bytes)))


def array_summary(dataset):
    digest = hashlib.sha256()
    nonfinite = 0
    step = block_length(dataset)
    for start in range(0, dataset.shape[0], step):
        block = np.ascontiguousarray(dataset[start:start + step])
        digest.update(block.tobytes())
        if block.dtype.kind == "f":
            nonfinite += int(block.size - np.count_nonzero(np.isfinite(block)))
    return {"values_sha256": digest.hexdigest(), "nonfinite_values": nonfinite}


def reviewed_attributes(group):
    """Retain the controller JSON found in the audited source trajectories."""
    if set(group.attrs) - {"config"}:
        raise ValueError(f"Review source attributes before export: {group.name}")
    attributes = {}
    if "config" in group.attrs:
        value = group.attrs["config"]
        text = value.decode("utf-8") if isinstance(value, bytes) else value
        if not isinstance(text, str):
            raise ValueError(f"Expected controller JSON text: {group.name}")
        config = json.loads(text)
        if not isinstance(config, dict) or set(config) != {"controller_cfg", "controller_type"}:
            raise ValueError(f"Review controller config before export: {group.name}")
        attributes["config"] = {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "encoding": "utf-8",
        }
    return attributes


def export_episode(arguments):
    source, identity, group_name, output, compression = arguments
    if source_identity(source) != identity:
        raise RuntimeError("Source dataset changed during export")
    output = Path(output)
    destination = output / "episodes" / f"{group_name}.hdf5"
    record_path = output / ".records" / f"{group_name}.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        if destination.is_file() and sha256_file(destination) == record["sha256"]:
            return record
        raise RuntimeError(f"Previously exported episode changed: {destination}")

    temporary = None
    try:
        with h5py.File(source, "r") as source_file:
            group = source_file[group_name]
            names = set(group)
            if not REQUIRED_FIELDS <= names or names - REQUIRED_FIELDS - OPTIONAL_FIELDS:
                raise ValueError(f"Unexpected fields in {group_name}: {sorted(names)}")
            attributes = reviewed_attributes(group)
            if any(group[name].attrs for name in names):
                raise ValueError(f"Review source attributes before export: {group_name}")
            frames = group["actions"].shape[0]
            fields = {}
            for name in sorted(names):
                dataset = group[name]
                if not isinstance(dataset, h5py.Dataset) or dataset.dtype.kind not in "biuf":
                    raise ValueError(f"Expected a numeric dataset: {group_name}/{name}")
                allowed_lengths = (
                    {frames, frames + 1} if name in TERMINAL_OBSERVATION_FIELDS else {frames}
                )
                if not dataset.shape or dataset.shape[0] not in allowed_lengths or frames == 0:
                    raise ValueError(f"Misaligned time dimension: {group_name}/{name}")
                fields[name] = {
                    "shape": list(dataset.shape), "dtype": str(dataset.dtype),
                    "values_sha256": None, "nonfinite_values": 0,
                }

            # Recover an episode finished just before an interrupted manifest write.
            if destination.exists():
                for name in sorted(names):
                    fields[name].update(array_summary(group[name]))
            else:
                descriptor, temporary = tempfile.mkstemp(
                    prefix=group_name + "-", suffix=".partial", dir=destination.parent,
                )
                os.close(descriptor)
                with h5py.File(temporary, "w") as target:
                    data = target.create_group("data")
                    data.attrs["original_trajectory_id"] = group_name
                    for name in attributes:
                        data.attrs.create(
                            name, group.attrs[name], dtype=group.attrs.get_id(name).dtype,
                        )
                    for name in sorted(names):
                        dataset = group[name]
                        options = {
                            "compression": compression, "shuffle": True, "fletcher32": True,
                            "chunks": (min(dataset.shape[0], block_length(dataset)), *dataset.shape[1:]),
                        }
                        if compression == "gzip":
                            options["compression_opts"] = 1
                        copied = data.create_dataset(
                            name, shape=dataset.shape, dtype=dataset.dtype, **options,
                        )
                        digest = hashlib.sha256()
                        step = block_length(dataset)
                        for start in range(0, dataset.shape[0], step):
                            block = np.ascontiguousarray(dataset[start:start + step])
                            copied[start:start + step] = block
                            digest.update(block.tobytes())
                            if block.dtype.kind == "f":
                                fields[name]["nonfinite_values"] += int(
                                    block.size - np.count_nonzero(np.isfinite(block))
                                )
                        fields[name]["values_sha256"] = digest.hexdigest()

            verified_path = destination if destination.exists() else Path(temporary)
            with h5py.File(verified_path, "r") as target:
                data = target["data"]
                if set(data) != names:
                    raise RuntimeError(f"Exported fields differ: {group_name}")
                if set(data.attrs) != set(attributes) | {"original_trajectory_id"}:
                    raise RuntimeError(f"Exported attributes differ: {group_name}")
                if data.attrs["original_trajectory_id"] != group_name:
                    raise RuntimeError(f"Exported trajectory ID differs: {group_name}")
                for name in attributes:
                    if (data.attrs[name] != group.attrs[name]
                            or data.attrs.get_id(name).dtype != group.attrs.get_id(name).dtype):
                        raise RuntimeError(f"Attribute verification failed: {group_name}/{name}")
                for name, description in fields.items():
                    copied = data[name]
                    if (list(copied.shape) != description["shape"]
                            or str(copied.dtype) != description["dtype"]
                            or array_summary(copied)["values_sha256"] != description["values_sha256"]):
                        raise RuntimeError(f"Lossless verification failed: {group_name}/{name}")
            if source_identity(source) != identity:
                raise RuntimeError("Source dataset changed during export")
            if temporary is not None:
                os.replace(temporary, destination)
                temporary = None
        record = {
            "trajectory_id": group_name,
            "file": str(destination.relative_to(output)),
            "frames": frames, "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination), "fields": fields,
            "attributes": attributes,
        }
        write_json(record_path, record)
        return record
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def export_dataset(source, normalization, output, *, task, compression="lzf",
                   seed=42, num_test=50, workers=2, limit=None):
    source = Path(source).resolve()
    normalization = Path(normalization).resolve()
    output = Path(output).resolve()
    identity = source_identity(source)
    normalization_bytes = normalization.read_bytes()
    stats = json.loads(normalization_bytes)
    if not {"mean_acs", "std_acs"} <= set(stats):
        raise ValueError("The selected recipe requires action mean/std statistics")
    with h5py.File(source, "r") as handle:
        if handle.attrs:
            raise ValueError("Review source root attributes before export")
        trajectory_ids = sorted(handle)
        if not trajectory_ids or any(
            not name.startswith("trajectory_") or "/" in name or "\\" in name
            for name in trajectory_ids
        ):
            raise ValueError("Expected only trajectory_* groups in the source file")
    if not 0 < num_test < len(trajectory_ids):
        raise ValueError("num_test must leave at least one training and one test episode")
    state = {
        "format_version": 1, "source": identity, "task": task,
        "normalization_sha256": hashlib.sha256(normalization_bytes).hexdigest(),
        "compression": compression, "seed": seed, "num_test": num_test,
    }
    state_path = output / ".export-state.json"
    if state_path.exists():
        if json.loads(state_path.read_text()) != state:
            raise ValueError("Output belongs to a different source or export configuration")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to use an unrecognized nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "episodes").mkdir(exist_ok=True)
    (output / ".records").mkdir(exist_ok=True)
    write_json(state_path, state)
    (output / "normalization.json").write_bytes(normalization_bytes)
    shuffled = trajectory_ids.copy()
    random.Random(seed).shuffle(shuffled)
    splits = {
        "seed": seed, "num_test": num_test,
        "train": shuffled[num_test:], "test": shuffled[:num_test],
    }
    write_json(output / "splits.json", splits)
    selected = trajectory_ids if limit is None else trajectory_ids[:limit]
    arguments = [(str(source), identity, name, str(output), compression) for name in selected]
    records = []
    if workers == 1:
        for argument in arguments:
            record = export_episode(argument)
            records.append(record)
            print(f"{task}: {len(records)}/{len(selected)} {record['trajectory_id']}", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = [executor.submit(export_episode, argument) for argument in arguments]
            try:
                for future in as_completed(futures):
                    record = future.result()
                    records.append(record)
                    print(f"{task}: {len(records)}/{len(selected)} {record['trajectory_id']}", flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    records.sort(key=lambda item: item["trajectory_id"])
    manifest = output / "manifest.jsonl"
    temporary_manifest = output / "manifest.jsonl.tmp"
    temporary_manifest.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in records))
    os.replace(temporary_manifest, manifest)
    metadata = {
        "format_version": 1, "task": task,
        "complete": len(records) == len(trajectory_ids),
        "episodes": len(records), "expected_episodes": len(trajectory_ids),
        "frames": sum(item["frames"] for item in records),
        "export_bytes": sum(item["bytes"] for item in records),
        "source_dataset": source.name, "source_bytes": identity["bytes"],
        "compression": compression, "array_values_and_dtypes_preserved": True,
        "normalization_sha256": state["normalization_sha256"],
        "manifest_sha256": sha256_file(manifest),
        "nonfinite_values": sum(
            field["nonfinite_values"] for item in records for field in item["fields"].values()
        ),
        "episodes_with_terminal_observations": sum(
            any(field["shape"][0] == item["frames"] + 1 for field in item["fields"].values())
            for item in records
        ),
        "episodes_with_controller_config": sum(
            "config" in item.get("attributes", {}) for item in records
        ),
        "paper": "https://arxiv.org/abs/2606.14981",
    }
    write_json(output / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", required=True, choices=["pipette", "insertion", "wiping"])
    parser.add_argument("--compression", choices=["lzf", "gzip"], default="lzf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-test", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Export only this many episodes for a local smoke test")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers and --limit must be positive")
    export_dataset(
        args.source, args.normalization, args.output, task=args.task,
        compression=args.compression, seed=args.seed, num_test=args.num_test,
        workers=args.workers, limit=args.limit,
    )


if __name__ == "__main__":
    main()

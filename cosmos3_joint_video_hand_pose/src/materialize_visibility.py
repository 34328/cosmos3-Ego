"""Materialize per-frame palm-in-FOV labels required by the EgoVerse loss."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import zarr


SIDES = ("left", "right")
FIELD_TEMPLATE = "{side}.obs_palm_in_fov_front_1"
VERSION = "palm_in_fov_front_1_source_640x360_v1"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360


def project_palm_visibility(
    head_pose: np.ndarray,
    palm_world: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    """Project SLAM-world palm centers through each frame's head camera."""
    head_pose = np.asarray(head_pose, dtype=np.float64)
    palm_world = np.asarray(palm_world, dtype=np.float64)
    projection = np.asarray(projection, dtype=np.float64)
    frames = len(head_pose)
    if head_pose.shape != (frames, 7):
        raise ValueError(f"obs_head_pose has shape {head_pose.shape}, expected {(frames, 7)}")
    if palm_world.shape != (frames, 3):
        raise ValueError(f"palm positions have shape {palm_world.shape}, expected {(frames, 3)}")
    if projection.shape != (3, 4):
        raise ValueError(f"intrinsics.front_1 has shape {projection.shape}, expected (3, 4)")

    # Stored quaternion order is [qw, qx, qy, qz]; scipy expects [qx, qy, qz, qw].
    rotation_world_from_camera = Rotation.from_quat(head_pose[:, [4, 5, 6, 3]]).as_matrix()
    palm_camera = np.einsum(
        "tji,tj->ti",
        rotation_world_from_camera,
        palm_world - head_pose[:, :3],
    )
    homogeneous = np.concatenate([palm_camera, np.ones((frames, 1), dtype=np.float64)], axis=1)
    projected = homogeneous @ projection.T
    denominator = projected[:, 2]
    safe_denominator = np.where(np.abs(denominator) > 1e-12, denominator, 1.0)
    u = projected[:, 0] / safe_denominator
    v = projected[:, 1] / safe_denominator
    finite = (
        np.isfinite(palm_camera).all(axis=1)
        & np.isfinite(projected).all(axis=1)
        & np.isfinite(u)
        & np.isfinite(v)
    )
    return (
        finite
        & (palm_camera[:, 2] > 0)
        & (np.abs(denominator) > 1e-12)
        & (u >= 0)
        & (u < IMAGE_WIDTH)
        & (v >= 0)
        & (v < IMAGE_HEIGHT)
    ).astype(np.uint8)


def compute_visibility(group: zarr.Group, total_frames: int, side: str) -> np.ndarray:
    return project_palm_visibility(
        np.asarray(group["obs_head_pose"][:total_frames], dtype=np.float64),
        np.asarray(group[f"{side}.obs_ee_pose"][:total_frames, :3], dtype=np.float64),
        np.asarray(group.attrs["intrinsics"]["front_1"], dtype=np.float64),
    )


def field_is_complete(array: zarr.Array, total_frames: int) -> bool:
    return (
        array.shape == (total_frames,)
        and array.dtype == np.dtype(np.uint8)
        and array.attrs.get("derivation_version") == VERSION
        and array.attrs.get("complete") is True
    )


def write_field(group: zarr.Group, name: str, values: np.ndarray) -> None:
    if name in group:
        del group[name]
    array = group.create_array(
        name,
        data=values,
        chunks=(min(4096, len(values)),),
        dimension_names=("frame",),
        attributes={
            "complete": False,
            "derivation_version": VERSION,
            "definition": "GT palm center projects inside original images.front_1 and z_camera > 0",
            "source_image_key": "images.front_1",
            "source_image_width": IMAGE_WIDTH,
            "source_image_height": IMAGE_HEIGHT,
            "source_palm_key": name.replace("obs_palm_in_fov_front_1", "obs_ee_pose"),
            "source_camera_pose_key": "obs_head_pose",
            "source_intrinsics_attr": "intrinsics.front_1",
        },
    )
    if not np.array_equal(np.asarray(array[:], dtype=np.uint8), values):
        del group[name]
        raise RuntimeError(f"readback mismatch for {name}")
    array.attrs.update(
        visible_count=int(values.sum()),
        out_of_fov_count=int(len(values) - values.sum()),
        complete=True,
    )


def update_root_metadata(group: zarr.Group, total_frames: int) -> None:
    features = dict(group.attrs.get("features", {}))
    fields = []
    for side in SIDES:
        name = FIELD_TEMPLATE.format(side=side)
        fields.append(name)
        features[name] = {
            "dtype": "uint8",
            "shape": [],
            "names": [],
            "semantics": "GT palm center in original front_1 camera FOV; not an occlusion label",
        }
    group.attrs["features"] = features
    derived = dict(group.attrs.get("derived_annotations", {}))
    derived["palm_in_fov_front_1"] = {
        "complete": True,
        "version": VERSION,
        "total_frames": total_frames,
        "image_size_wh": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "fields": fields,
    }
    group.attrs["derived_annotations"] = derived


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.episodes_manifest.open(newline="", encoding="utf-8") as handle:
        episodes = list(csv.DictReader(handle))
    totals = {side: {"frames": 0, "visible": 0} for side in SIDES}
    created = reused = 0
    episode_rows = []
    for episode in episodes:
        group = zarr.open_group(episode["abs_zarr_path"], mode="r" if args.dry_run else "a")
        total_frames = int(group.attrs["total_frames"])
        if total_frames != int(episode["total_frames"]):
            raise ValueError(f"{episode['episode_hash']}: manifest/Zarr total_frames mismatch")
        row = {"episode_hash": episode["episode_hash"], "total_frames": total_frames}
        wrote_episode = False
        for side in SIDES:
            values = compute_visibility(group, total_frames, side)
            name = FIELD_TEMPLATE.format(side=side)
            matches = False
            if name in group and field_is_complete(group[name], total_frames):
                matches = np.array_equal(np.asarray(group[name][:], dtype=np.uint8), values)
            if name in group and not matches and not (args.overwrite or args.dry_run):
                raise RuntimeError(f"{episode['episode_hash']}: existing {name} differs from recomputation")
            if not args.dry_run and not matches:
                write_field(group, name, values)
                wrote_episode = True
            row[f"{side}_visible"] = int(values.sum())
            row[f"{side}_out_of_fov"] = int(total_frames - values.sum())
            totals[side]["frames"] += total_frames
            totals[side]["visible"] += int(values.sum())
        if not args.dry_run:
            update_root_metadata(group, total_frames)
        created += int(wrote_episode)
        reused += int(not wrote_episode and all(FIELD_TEMPLATE.format(side=s) in group for s in SIDES))
        episode_rows.append(row)

    report = {
        "status": "dry_run_pass" if args.dry_run else "pass",
        "version": VERSION,
        "episodes_manifest": str(args.episodes_manifest.resolve()),
        "episodes_manifest_sha256": hashlib.sha256(args.episodes_manifest.read_bytes()).hexdigest(),
        "episode_count": len(episodes),
        "materialized_episode_count": created,
        "reused_episode_count": reused,
        "visibility": {
            side: {
                **counts,
                "out_of_fov": counts["frames"] - counts["visible"],
                "visible_fraction": counts["visible"] / counts["frames"],
            }
            for side, counts in totals.items()
        },
        "episodes": episode_rows,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()

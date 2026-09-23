#!/usr/bin/env python3
"""Quantify the geometric applicability domain of the frozen BG training set.

The analysis uses only positions, cells, structure identifiers, and the exact
seed-42 split.  Source scientific assets are read-only.  Bond topology and the
two layer surfaces are reconstructed independently in every stored structure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay


ROOT = Path(__file__).resolve().parents[1]
from training_geometry_style import BLUE, INK, ORANGE, RED, TEAL, finish, panel_label, setup

DEFAULT_INPUT = ROOT / "doc" / "data" / "training_geometry" / "bg_graph_geometry.npz"
DEFAULT_OUTPUT = ROOT / "doc" / "output" / "training_geometry"
DEFAULT_FIGURES = DEFAULT_OUTPUT / "figures"
DEFAULT_MANIFEST = ROOT / "doc" / "data" / "training_geometry" / "seed42_test_manifest.json"
SOURCE_GRAPH = "HGraph-BilayerGrapheneTaskMet-rFromDFT-edge=Aij.pkl"
SOURCE_GRAPH_SHA256 = "66a5aacfc80ffe645c8c076168ebc69b0b68c3a4b632e9b3767f37df8077f09f"


QUANTILES = (0.0, 0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995, 1.0)
Q_NAMES = ("min", "q005", "q01", "q05", "median", "q95", "q99", "q995", "max")
SPLIT_COLORS = {"train": BLUE, "validation": ORANGE, "test": TEAL}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def natural_key(value: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", value)
    return tuple(int(x) for x in parts) if parts else (10**9,)


def frozen_split(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n != 300:
        raise RuntimeError(f"Expected 300 frozen structures, found {n}")
    indices = np.arange(n, dtype=np.int64)
    np.random.seed(seed)
    np.random.shuffle(indices)
    return indices[:180], indices[180:240], indices[240:300]


def validate_manifest(
    manifest_path: Path,
    test_indices: np.ndarray,
    structure_ids: np.ndarray,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registered_indices = np.asarray(manifest["test_indices"], dtype=np.int64)
    registered_ids = [str(x) for x in manifest["test_structure_ids"]]
    actual_ids = [str(structure_ids[i]) for i in test_indices]
    if not np.array_equal(test_indices, registered_indices):
        raise RuntimeError("Reproduced seed-42 test indices do not match the frozen manifest")
    if actual_ids != registered_ids:
        raise RuntimeError("Reproduced seed-42 test IDs do not match the frozen manifest")
    if manifest.get("graph_sha256") != SOURCE_GRAPH_SHA256:
        raise RuntimeError("Frozen manifest graph hash differs from the registered source graph")
    return manifest


def normal_from_cell(cell: np.ndarray) -> np.ndarray:
    normal = np.cross(cell[0], cell[1])
    norm = float(np.linalg.norm(normal))
    if norm <= 0:
        raise RuntimeError("Degenerate in-plane cell")
    return normal / norm


def mic_vectors(pos_a: np.ndarray, pos_b: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Return vectors pos_b-pos_a under the three-dimensional minimum image."""
    delta = pos_b[None, :, :] - pos_a[:, None, :]
    frac = delta @ np.linalg.inv(cell)
    frac -= np.rint(frac)
    return frac @ cell


def choose_layers(reference_positions: np.ndarray, reference_cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normal_from_cell(reference_cell)
    heights = reference_positions @ normal
    order = np.argsort(heights)
    gaps = np.diff(heights[order])
    split = int(np.argmax(gaps)) + 1
    if split != len(reference_positions) // 2:
        raise RuntimeError(f"Layer partition is not 32/32; split={split}")
    lower = np.sort(order[:split])
    upper = np.sort(order[split:])
    if float(gaps[split - 1]) < 1.0:
        raise RuntimeError("No clear bilayer height gap in the reference structure")
    return lower, upper


def reconstruct_bond_topology(
    reference_positions: np.ndarray,
    reference_cell: np.ndarray,
    layer_groups: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, float, dict]:
    third_distances = []
    fourth_distances = []
    for group in layer_groups:
        vectors = mic_vectors(reference_positions[group], reference_positions[group], reference_cell)
        distances = np.linalg.norm(vectors, axis=-1)
        np.fill_diagonal(distances, np.inf)
        ordered = np.sort(distances, axis=1)
        third_distances.extend(ordered[:, 2].tolist())
        fourth_distances.extend(ordered[:, 3].tolist())
    lower_bound = float(max(third_distances))
    upper_bound = float(min(fourth_distances))
    if not lower_bound < upper_bound:
        raise RuntimeError(
            f"No clean 3rd/4th-neighbor topology gap: {lower_bound} vs {upper_bound}"
        )
    cutoff = 0.5 * (lower_bound + upper_bound)

    pairs: list[tuple[int, int]] = []
    degrees = np.zeros(len(reference_positions), dtype=int)
    for group in layer_groups:
        vectors = mic_vectors(reference_positions[group], reference_positions[group], reference_cell)
        distances = np.linalg.norm(vectors, axis=-1)
        for local_i in range(len(group)):
            for local_j in range(local_i + 1, len(group)):
                if distances[local_i, local_j] < cutoff:
                    i = int(group[local_i])
                    j = int(group[local_j])
                    pairs.append((i, j))
                    degrees[i] += 1
                    degrees[j] += 1
    pair_array = np.asarray(pairs, dtype=np.int64)
    if pair_array.shape != (96, 2) or not np.all(degrees == 3):
        raise RuntimeError(
            f"Unexpected graphene bond topology: pairs={pair_array.shape}, "
            f"degrees={np.unique(degrees, return_counts=True)}"
        )
    audit = {
        "third_neighbor_max_angstrom": lower_bound,
        "fourth_neighbor_min_angstrom": upper_bound,
        "derived_bond_cutoff_angstrom": cutoff,
        "bond_count": int(len(pair_array)),
        "coordination": 3,
    }
    return pair_array, cutoff, audit


def pair_distances(positions: np.ndarray, cell: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    delta = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    frac = delta @ np.linalg.inv(cell)
    frac -= np.rint(frac)
    return np.linalg.norm(frac @ cell, axis=1)


def periodic_surface_heights(
    query_positions: np.ndarray,
    surface_positions: np.ndarray,
    cell: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    """Interpolate a periodic piecewise-linear atomic surface at query in-plane points."""
    e1 = cell[0] / np.linalg.norm(cell[0])
    e2 = np.cross(normal, e1)
    e2 /= np.linalg.norm(e2)
    inplane_cell = np.asarray(
        [
            [cell[0] @ e1, cell[0] @ e2],
            [cell[1] @ e1, cell[1] @ e2],
        ]
    )
    inverse_inplane_cell = np.linalg.inv(inplane_cell)

    query_xy = np.column_stack((query_positions @ e1, query_positions @ e2))
    surface_xy = np.column_stack((surface_positions @ e1, surface_positions @ e2))
    query_fractional = query_xy @ inverse_inplane_cell
    surface_fractional = surface_xy @ inverse_inplane_cell
    query_fractional -= np.floor(query_fractional)
    surface_fractional -= np.floor(surface_fractional)
    query_xy = query_fractional @ inplane_cell
    surface_xy = surface_fractional @ inplane_cell

    image_points = []
    image_heights = []
    surface_heights = surface_positions @ normal
    for shift_a in (-1, 0, 1):
        for shift_b in (-1, 0, 1):
            shift = shift_a * inplane_cell[0] + shift_b * inplane_cell[1]
            image_points.append(surface_xy + shift)
            image_heights.append(surface_heights)
    image_points_array = np.concatenate(image_points, axis=0)
    image_heights_array = np.concatenate(image_heights, axis=0)

    triangulation = Delaunay(image_points_array)
    simplices = triangulation.find_simplex(query_xy)
    if np.any(simplices < 0):
        raise RuntimeError("Periodic surface triangulation did not cover all query points")
    affine = triangulation.transform[simplices, :2]
    offsets = query_xy - triangulation.transform[simplices, 2]
    first_two_weights = np.einsum("nij,nj->ni", affine, offsets)
    barycentric = np.column_stack(
        (first_two_weights, 1.0 - np.sum(first_two_weights, axis=1))
    )
    vertices = triangulation.simplices[simplices]
    return np.sum(image_heights_array[vertices] * barycentric, axis=1)


def periodic_surface_spacings(
    positions: np.ndarray,
    cell: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    """Evaluate the two periodic layer surfaces at identical in-plane positions."""
    upper_at_lower = periodic_surface_heights(
        positions[lower], positions[upper], cell, normal
    )
    lower_at_upper = periodic_surface_heights(
        positions[upper], positions[lower], cell, normal
    )
    lower_heights = positions[lower] @ normal
    upper_heights = positions[upper] @ normal
    return np.concatenate(
        (np.abs(upper_at_lower - lower_heights), np.abs(upper_heights - lower_at_upper))
    )


def qdict(prefix: str, values: np.ndarray) -> dict[str, float]:
    q = np.quantile(np.asarray(values, dtype=float), QUANTILES)
    result = {f"{prefix}_{name}": float(value) for name, value in zip(Q_NAMES, q)}
    result[f"{prefix}_mean"] = float(np.mean(values))
    result[f"{prefix}_std"] = float(np.std(values, ddof=0))
    return result


def frame_metrics(
    positions: np.ndarray,
    cell: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    lower, upper = choose_layers(positions, cell)
    bond_pairs, bond_cutoff, topology_audit = reconstruct_bond_topology(
        positions, cell, (lower, upper)
    )
    normal = normal_from_cell(cell)
    lower_h = positions[lower] @ normal
    upper_h = positions[upper] @ normal
    if float(np.mean(lower_h)) > float(np.mean(upper_h)):
        lower_h, upper_h = upper_h, lower_h
        lower, upper = upper, lower
    separation_gap = float(np.min(upper_h) - np.max(lower_h))
    if separation_gap <= 1.0:
        raise RuntimeError(f"Layer labels crossed or collapsed; projected gap={separation_gap}")

    bonds = pair_distances(positions, cell, bond_pairs)
    cross_vectors = mic_vectors(positions[lower], positions[upper], cell)
    total = np.linalg.norm(cross_vectors, axis=-1)

    local_spacing = periodic_surface_spacings(positions, cell, lower, upper, normal)
    nearest_3d = np.concatenate([np.min(total, axis=1), np.min(total, axis=0)])

    lower_centered = lower_h - np.mean(lower_h)
    upper_centered = upper_h - np.mean(upper_h)
    corrugation_rms_layers = np.asarray(
        [np.sqrt(np.mean(lower_centered**2)), np.sqrt(np.mean(upper_centered**2))]
    )
    corrugation_ptp_layers = np.asarray([np.ptp(lower_h), np.ptp(upper_h)])
    mean_layer_spacing = float(np.mean(upper_h) - np.mean(lower_h))
    inplane_area = float(np.linalg.norm(np.cross(cell[0], cell[1])))

    metrics: dict[str, float] = {}
    metrics.update(qdict("bond", bonds))
    metrics.update(qdict("local_layer_spacing", local_spacing))
    metrics.update(qdict("interlayer_nn3d", nearest_3d))
    metrics.update(
        {
            "mean_layer_spacing_angstrom": mean_layer_spacing,
            "layer_gap_angstrom": separation_gap,
            "corrugation_rms_mean_angstrom": float(np.mean(corrugation_rms_layers)),
            "corrugation_rms_max_angstrom": float(np.max(corrugation_rms_layers)),
            "corrugation_ptp_mean_angstrom": float(np.mean(corrugation_ptp_layers)),
            "corrugation_ptp_max_angstrom": float(np.max(corrugation_ptp_layers)),
            "inplane_area_angstrom2": inplane_area,
            "adaptive_bond_cutoff_angstrom": bond_cutoff,
            "third_neighbor_max_angstrom": topology_audit["third_neighbor_max_angstrom"],
            "fourth_neighbor_min_angstrom": topology_audit["fourth_neighbor_min_angstrom"],
        }
    )
    samples = {
        "bond_length_angstrom": bonds,
        "local_layer_spacing_angstrom": local_spacing,
        "interlayer_nearest_3d_angstrom": nearest_3d,
    }
    return metrics, samples


def axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Return a deterministic proper rotation matrix."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_minus_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ]
    )


def covalent_radius_topology_audit(
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
) -> dict[str, int | bool]:
    """Independently audit C-C topology using a fixed covalent-radius cutoff."""
    groups = choose_layers(positions, cell)
    degrees = np.zeros(len(positions), dtype=int)
    pair_count = 0
    for group in groups:
        distances = np.linalg.norm(mic_vectors(positions[group], positions[group], cell), axis=-1)
        for local_i in range(len(group)):
            for local_j in range(local_i + 1, len(group)):
                if distances[local_i, local_j] < cutoff:
                    degrees[int(group[local_i])] += 1
                    degrees[int(group[local_j])] += 1
                    pair_count += 1
    return {
        "pair_count": pair_count,
        "all_atoms_degree_three": bool(np.all(degrees == 3)),
        "minimum_degree": int(np.min(degrees)),
        "maximum_degree": int(np.max(degrees)),
    }


def run_geometry_qa(positions: np.ndarray, lattices: np.ndarray) -> dict:
    """Run definition-level QA independent of the statistical envelopes."""
    rotation = axis_angle_rotation(np.asarray([1.7, -0.8, 2.3]), angle_rad=0.731)
    rotation_columns = rotation.T
    compared_keys = (
        "bond_min",
        "bond_max",
        "local_layer_spacing_min",
        "local_layer_spacing_max",
        "interlayer_nn3d_min",
        "interlayer_nn3d_max",
        "mean_layer_spacing_angstrom",
        "corrugation_ptp_max_angstrom",
    )
    maximum_rotation_error = 0.0
    maximum_sample_rotation_error = 0.0
    covalent_cutoff = 1.2 * (0.76 + 0.76)
    covalent_failures: list[int] = []
    minimum_neighbor_gap = math.inf

    for frame_index, (frame_positions, frame_cell) in enumerate(zip(positions, lattices)):
        original_metrics, original_samples = frame_metrics(frame_positions, frame_cell)
        rotated_metrics, rotated_samples = frame_metrics(
            frame_positions @ rotation_columns,
            frame_cell @ rotation_columns,
        )
        maximum_rotation_error = max(
            maximum_rotation_error,
            max(abs(original_metrics[key] - rotated_metrics[key]) for key in compared_keys),
        )
        maximum_sample_rotation_error = max(
            maximum_sample_rotation_error,
            max(
                float(np.max(np.abs(original_samples[key] - rotated_samples[key])))
                for key in original_samples
            ),
        )
        covalent_audit = covalent_radius_topology_audit(
            frame_positions, frame_cell, covalent_cutoff
        )
        if covalent_audit["pair_count"] != 96 or not covalent_audit["all_atoms_degree_three"]:
            covalent_failures.append(frame_index)
        minimum_neighbor_gap = min(
            minimum_neighbor_gap,
            original_metrics["fourth_neighbor_min_angstrom"]
            - original_metrics["third_neighbor_max_angstrom"],
        )

    normal = np.asarray([0.0, 0.0, 1.0])
    perpendicular_gap = 3.35
    lateral_shift = np.asarray([1.42, -0.77, 0.0])
    vertical_vector = perpendicular_gap * normal
    oblique_vector = vertical_vector + lateral_shift
    projected_vertical = float(abs(vertical_vector @ normal))
    projected_oblique = float(abs(oblique_vector @ normal))
    distance_vertical = float(np.linalg.norm(vertical_vector))
    distance_oblique = float(np.linalg.norm(oblique_vector))
    synthetic_cell = np.asarray(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 20.0]]
    )
    synthetic_lower = np.asarray(
        [[0.25, 0.25, 0.0], [1.25, 0.25, 0.0], [0.25, 1.25, 0.0], [1.25, 1.25, 0.0]]
    )
    synthetic_upper = synthetic_lower + np.asarray([0.40, 0.30, perpendicular_gap])
    synthetic_positions = np.vstack((synthetic_lower, synthetic_upper))
    synthetic_spacings = periodic_surface_spacings(
        synthetic_positions,
        synthetic_cell,
        np.arange(4, dtype=int),
        np.arange(4, 8, dtype=int),
        normal,
    )
    maximum_surface_interpolation_error = float(
        np.max(np.abs(synthetic_spacings - perpendicular_gap))
    )

    rotation_tolerance = 1.0e-10
    qa = {
        "rigid_rotation_invariance": {
            "frames_tested": int(len(positions)),
            "rotation_axis": [1.7, -0.8, 2.3],
            "rotation_angle_rad": 0.731,
            "maximum_metric_absolute_error_angstrom": maximum_rotation_error,
            "maximum_raw_sample_absolute_error_angstrom": maximum_sample_rotation_error,
            "tolerance_angstrom": rotation_tolerance,
            "passed": bool(
                maximum_rotation_error <= rotation_tolerance
                and maximum_sample_rotation_error <= rotation_tolerance
            ),
        },
        "synthetic_oblique_vector": {
            "perpendicular_gap_angstrom": perpendicular_gap,
            "lateral_shift_angstrom": float(np.linalg.norm(lateral_shift)),
            "normal_projection_without_lateral_shift_angstrom": projected_vertical,
            "normal_projection_with_lateral_shift_angstrom": projected_oblique,
            "distance_without_lateral_shift_angstrom": distance_vertical,
            "distance_with_lateral_shift_angstrom": distance_oblique,
            "periodic_surface_interpolation_max_error_angstrom": maximum_surface_interpolation_error,
            "projection_invariant": bool(abs(projected_vertical - projected_oblique) <= 1.0e-12),
            "three_dimensional_distance_changes": bool(distance_oblique > distance_vertical),
            "passed": bool(
                abs(projected_vertical - projected_oblique) <= 1.0e-12
                and distance_oblique > distance_vertical
                and maximum_surface_interpolation_error <= 1.0e-12
            ),
        },
        "independent_covalent_radius_topology": {
            "carbon_covalent_radius_angstrom": 0.76,
            "scale_factor": 1.2,
            "cutoff_angstrom": covalent_cutoff,
            "frames_tested": int(len(positions)),
            "failed_frame_indices": covalent_failures,
            "all_frames_have_96_bonds_and_degree_three": not covalent_failures,
            "passed": not covalent_failures,
        },
        "nearest_neighbor_separation": {
            "minimum_framewise_fourth_minus_third_neighbor_gap_angstrom": minimum_neighbor_gap,
            "passed": bool(minimum_neighbor_gap > 0.0),
        },
    }
    qa["all_checks_passed"] = bool(
        qa["rigid_rotation_invariance"]["passed"]
        and qa["synthetic_oblique_vector"]["passed"]
        and qa["independent_covalent_radius_topology"]["passed"]
        and qa["nearest_neighbor_separation"]["passed"]
    )
    return qa


def envelope_row(metric: str, scope: str, values: np.ndarray, unit: str) -> dict[str, float | str | int]:
    values = np.asarray(values, dtype=float)
    q = np.quantile(values, QUANTILES)
    row: dict[str, float | str | int] = {
        "metric": metric,
        "scope": scope,
        "unit": unit,
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std_population": float(np.std(values, ddof=0)),
    }
    row.update({name: float(value) for name, value in zip(Q_NAMES, q)})
    return row


def convex_hull(points: np.ndarray) -> np.ndarray:
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(points) <= 2:
        return points
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ao = a - o
        bo = b - o
        return float(ao[0] * bo[1] - ao[1] * bo[0])

    lower: list[np.ndarray] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in points[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def plot_envelopes(frame: pd.DataFrame, output_dir: Path) -> None:
    train = frame.loc[frame["split"] == "train"].copy()
    train["natural_key"] = train["structure_id"].map(natural_key)
    train = train.sort_values("natural_key").reset_index(drop=True)
    x = np.arange(len(train))

    setup()
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 11.7), constrained_layout=True)
    panels = [
        ("bond", "Intralayer C-C bond length (Å)", True),
        ("local_layer_spacing", "Local perpendicular layer spacing (Å)", True),
        ("interlayer_nn3d", "Nearest interlayer C-C distance (Å)", True),
    ]
    for panel_index, (prefix, ylabel, distributed) in enumerate(panels):
        axis = axes.flat[panel_index]
        minimum = train[f"{prefix}_min"].to_numpy(float)
        q05 = train[f"{prefix}_q05"].to_numpy(float)
        q95 = train[f"{prefix}_q95"].to_numpy(float)
        maximum = train[f"{prefix}_max"].to_numpy(float)
        axis.fill_between(x, minimum, maximum, color=BLUE, alpha=0.10, label="min-max")
        axis.fill_between(x, q05, q95, color=BLUE, alpha=0.28, label="5-95%")
        axis.plot(x, minimum, color=BLUE, lw=0.55)
        axis.plot(x, maximum, color=BLUE, lw=0.55)
        axis.set_ylabel(ylabel)
        axis.text(
            0.0,
            1.02,
            f"({chr(ord('a') + panel_index)})",
            transform=axis.transAxes,
            fontweight="bold",
            va="bottom",
            ha="left",
            fontsize=17.0,
        )
        axis.legend(frameon=False, loc="upper right", ncol=2)

    scalar_panels = [
        ("mean_layer_spacing_angstrom", "Mean plane-to-plane spacing (Å; diagnostic)"),
        ("corrugation_rms_mean_angstrom", "Mean layer corrugation RMS (Å; diagnostic)"),
        ("corrugation_ptp_max_angstrom", "Maximum layer peak-to-peak corrugation (Å)"),
    ]
    for offset, (column, ylabel) in enumerate(scalar_panels, start=3):
        axis = axes.flat[offset]
        values = train[column].to_numpy(float)
        q005, q995 = np.quantile(values, [0.005, 0.995])
        vmin, vmax = float(np.min(values)), float(np.max(values))
        axis.axhspan(vmin, vmax, color=BLUE, alpha=0.10, label="absolute envelope")
        axis.axhspan(q005, q995, color=BLUE, alpha=0.28, label="0.5-99.5%")
        axis.plot(x, values, color=BLUE, lw=0.9)
        axis.scatter(x, values, s=6, color=BLUE, edgecolors="none")
        axis.set_ylabel(ylabel)
        axis.text(
            0.0,
            1.02,
            f"({chr(ord('a') + offset)})",
            transform=axis.transAxes,
            fontweight="bold",
            va="bottom",
            ha="left",
            fontsize=17.0,
        )
        axis.legend(frameon=False, loc="upper right")

    for axis in axes[-1, :]:
        axis.set_xlabel("Training structure rank (natural structure-ID order)")
    finish(fig, output_dir / "figure_training_geometry_envelopes")


def plot_split_coverage(frame: pd.DataFrame, output_dir: Path) -> None:
    setup()
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.6), constrained_layout=True)
    panels = [
        ("bond_min", "Per-structure minimum C-C bond length (Å)"),
        ("bond_max", "Per-structure maximum C-C bond length (Å)"),
        ("local_layer_spacing_min", "Per-structure minimum perpendicular spacing (Å)"),
        ("local_layer_spacing_max", "Per-structure maximum perpendicular spacing (Å)"),
    ]
    for pidx, (column, xlabel) in enumerate(panels):
        axis = axes.flat[pidx]
        all_values = frame[column].to_numpy(float)
        bins = np.linspace(float(np.min(all_values)), float(np.max(all_values)), 32)
        for split in ("train", "validation", "test"):
            values = frame.loc[frame["split"] == split, column].to_numpy(float)
            axis.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                lw=1.25,
                color=SPLIT_COLORS[split],
                label=split.capitalize(),
            )
        train_values = frame.loc[frame["split"] == "train", column].to_numpy(float)
        q005, q995 = np.quantile(train_values, [0.005, 0.995])
        axis.axvspan(q005, q995, color=BLUE, alpha=0.10)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Density")
        panel_label(axis, chr(ord("a") + pidx))
        if pidx == 0:
            axis.legend(frameon=False, loc="upper right")
    finish(fig, output_dir / "figure_training_validation_test_coverage")


def plot_joint_domain(frame: pd.DataFrame, output_dir: Path) -> None:
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), constrained_layout=True)
    pairs = [
        (
            "bond_min",
            "bond_max",
            "Per-structure minimum C-C bond length (Å)",
            "Per-structure maximum C-C bond length (Å)",
        ),
        (
            "local_layer_spacing_min",
            "local_layer_spacing_max",
            "Per-structure minimum perpendicular spacing (Å)",
            "Per-structure maximum perpendicular spacing (Å)",
        ),
    ]
    for pidx, (xcol, ycol, xlabel, ylabel) in enumerate(pairs):
        axis = axes[pidx]
        train = frame.loc[frame["split"] == "train", [xcol, ycol]].to_numpy(float)
        hull = convex_hull(train)
        if len(hull) >= 3:
            closed = np.vstack([hull, hull[0]])
            axis.fill(closed[:, 0], closed[:, 1], color=BLUE, alpha=0.08)
            axis.plot(closed[:, 0], closed[:, 1], color=BLUE, lw=0.9, label="Training hard hull")
        for split in ("train", "validation", "test"):
            subset = frame.loc[frame["split"] == split]
            axis.scatter(
                subset[xcol],
                subset[ycol],
                s=14 if split == "train" else 20,
                facecolors=SPLIT_COLORS[split] if split == "train" else "none",
                edgecolors=SPLIT_COLORS[split],
                linewidths=0.55,
                alpha=0.65 if split == "train" else 0.9,
                label=split.capitalize(),
            )
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        panel_label(axis, chr(ord("a") + pidx))
        if pidx == 1:
            axis.legend(frameon=False, loc="best")
    finish(fig, output_dir / "figure_training_geometry_joint_domain")


def write_method_report(output_path: Path, metadata: dict, qa: dict) -> None:
    bounds = metadata["training_bounds"]
    bond = bounds["bond_length_angstrom"]
    spacing = bounds["local_layer_spacing_angstrom"]
    distance_3d = bounds["interlayer_nearest_3d_angstrom"]
    topology = metadata["per_frame_topology"]
    rotation = qa["rigid_rotation_invariance"]
    oblique = qa["synthetic_oblique_vector"]
    covalent = qa["independent_covalent_radius_topology"]
    lines = [
        "# 双层石墨烯训练集几何适用域：方法与验证",
        "",
        "## 结论",
        "",
        "结构不能用单个典型键长或单个典型层间距代表。本分析对每个 64 原子结构分别保留 96 条真实面内 C–C 键和 64 个局域垂直层间距，并以逐结构最小值—最大值及 5%—95% 区间形成包络曲线。中位数与均值仅用于诊断，不作为适用域边界。",
        "",
        "## 真实键的判定",
        "",
        "每个结构独立完成以下操作：先沿晶胞基面法向分成两个 32 原子层，再在每层内使用周期性最小镜像距离。对每个原子排序同层邻居距离，第三近邻属于石墨烯的三个 $sp^2$ 键，第四近邻已进入下一配位壳。只有当第三与第四近邻之间存在严格间隙时，才在该间隙中点设自适应截断。",
        "",
        f"300 个结构全部得到 96 条唯一键，且 64 个原子的度数都严格为 3。全数据最大第三近邻为 {topology['third_neighbor_max_over_frames_angstrom']:.6f} Å，最小第四近邻为 {topology['fourth_neighbor_min_over_frames_angstrom']:.6f} Å；逐帧最小壳层间隙仍为 {qa['nearest_neighbor_separation']['minimum_framewise_fourth_minus_third_neighbor_gap_angstrom']:.6f} Å。因此被选中的边与任意第二配位壳原子间距之间有清晰、可审计的断层。",
        "",
        f"独立交叉检验使用 C 共价半径 0.76 Å 和 1.2 倍半径和截断，即 {covalent['cutoff_angstrom']:.3f} Å；300 个结构同样全部得到 96 条键和三配位。固定参考帧的原子编号配对曾产生约 3.8 Å 的假键，已被拒绝；原因是独立存储结构的原子编号不保证保持同一条化学键。",
        "",
        "## 真实垂直层间距的判定",
        "",
        "基面法向定义为 $\\mathbf{n}=(\\mathbf{a}_1\\times\\mathbf{a}_2)/|\\mathbf{a}_1\\times\\mathbf{a}_2|$。每个原子层在面内周期性复制后，以 Delaunay 三角剖分构成分片线性原子表面。对下层每个原子的面内坐标，在上层表面的同一面内坐标插值高度；再反向对上层原子插值下层高度。两高度沿 $\\mathbf{n}$ 的差才计为局域垂直层间距，上下两个方向共 64 个样本。",
        "",
        "这一定义明确排除了斜向伪间距：三维最近原子距离 $|\\Delta\\mathbf{r}_{ij}|$ 单独保存为对照，绝不作为垂直层间距。两层平均高度差只是平均平面间距诊断；在存在褶皱时，适用域采用 64 个局域垂直间距的逐结构包络。",
        "",
        f"合成反例中，垂直间距保持 {oblique['perpendicular_gap_angstrom']:.3f} Å，加入 {oblique['lateral_shift_angstrom']:.3f} Å 面内位移后，法向投影仍为 {oblique['normal_projection_with_lateral_shift_angstrom']:.3f} Å，而三维距离增至 {oblique['distance_with_lateral_shift_angstrom']:.6f} Å。对横向错开的两张平坦周期表面，插值恢复垂直间距的最大误差为 {oblique['periodic_surface_interpolation_max_error_angstrom']:.3e} Å。该检验通过。",
        "",
        "## 训练集包络",
        "",
        "| 指标 | 训练集绝对包络 / Å | 训练集稳健包络（0.5%—99.5%）/ Å |",
        "|---|---:|---:|",
        f"| 真实面内 C–C 键长 | {bond['absolute_min']:.6f}—{bond['absolute_max']:.6f} | {bond['robust_q005']:.6f}—{bond['robust_q995']:.6f} |",
        f"| 局域垂直层间距 | {spacing['absolute_min']:.6f}—{spacing['absolute_max']:.6f} | {spacing['robust_q005']:.6f}—{spacing['robust_q995']:.6f} |",
        f"| 最近跨层三维原子距离（对照） | {distance_3d['absolute_min']:.6f}—{distance_3d['absolute_max']:.6f} | {distance_3d['robust_q005']:.6f}—{distance_3d['robust_q995']:.6f} |",
        "",
        "绝对包络用于判断是否超出训练样本已观测的硬范围；稳健包络用于减少极端热涨落样本对边界的支配。两者都来自训练划分内的全部局域样本，而不是结构均值。逐结构包络曲线的数据另存为 `training_geometry_per_structure_envelopes.csv`。",
        "",
        "## 几何 QA",
        "",
        f"- 刚体旋转不变性：对全部 300 个结构同时旋转坐标和晶胞，最大指标误差为 {rotation['maximum_metric_absolute_error_angstrom']:.3e} Å，最大原始样本误差为 {rotation['maximum_raw_sample_absolute_error_angstrom']:.3e} Å；容差为 {rotation['tolerance_angstrom']:.1e} Å，检验通过。",
        "",
        "- 斜向位移反例：法向投影不变而三维距离改变，检验通过。",
        "",
        "- 独立共价半径拓扑：300/300 个结构均为 96 条键、全原子三配位，检验通过。",
        "",
        "- 总体结果：全部几何定义级 QA 通过。",
        "",
        "## 适用范围",
        "",
        "这些包络是必要的几何适用域，不是模型泛化的充分条件。它不能单独证明模型适用于扭转层、强曲率、拓扑改变、碳纳米管或富勒烯。结构 ID 仅按自然顺序作图；在缺少独立轨迹元数据时，不把该顺序解释为真实 MD 时间。",
        "",
    ]
    report = "\n".join(lines)
    without_display = report.replace("$$", "")
    if report.count("$$") % 2 or without_display.count("$") % 2:
        raise RuntimeError("Unbalanced Markdown math delimiters in method report")
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    arrays = np.load(args.input, allow_pickle=False)
    positions = np.asarray(arrays["positions_angstrom"], dtype=float)
    lattices = np.asarray(arrays["lattices_angstrom"], dtype=float)
    structure_ids = np.asarray(arrays["structure_ids"]).astype(str)
    dataset_indices = np.asarray(arrays["dataset_indices"], dtype=np.int64)
    if positions.shape != (300, 64, 3) or lattices.shape != (300, 3, 3):
        raise RuntimeError(f"Unexpected geometry shapes: {positions.shape}, {lattices.shape}")
    if not np.array_equal(dataset_indices, np.arange(300)):
        raise RuntimeError("Dataset indices are not the frozen 0..299 order")

    train_idx, val_idx, test_idx = frozen_split(len(structure_ids), seed=42)
    manifest = validate_manifest(args.manifest, test_idx, structure_ids)
    split = np.full(len(structure_ids), "", dtype="U16")
    split[train_idx] = "train"
    split[val_idx] = "validation"
    split[test_idx] = "test"

    reference_candidates = np.flatnonzero(structure_ids == "0_0")
    if len(reference_candidates) != 1:
        raise RuntimeError("Could not resolve unique reference structure 0_0")
    reference_index = int(reference_candidates[0])
    reference_lower, reference_upper = choose_layers(
        positions[reference_index], lattices[reference_index]
    )
    _, reference_bond_cutoff, reference_topology_audit = reconstruct_bond_topology(
        positions[reference_index],
        lattices[reference_index],
        (reference_lower, reference_upper),
    )

    rows: list[dict] = []
    samples_by_split: dict[str, dict[str, list[np.ndarray]]] = {
        name: {
            "bond_length_angstrom": [],
            "local_layer_spacing_angstrom": [],
            "interlayer_nearest_3d_angstrom": [],
        }
        for name in ("train", "validation", "test")
    }
    all_frame_samples: dict[str, list[np.ndarray]] = {
        "bond_length_angstrom": [],
        "local_layer_spacing_angstrom": [],
        "interlayer_nearest_3d_angstrom": [],
    }

    for index in range(len(structure_ids)):
        metrics, samples = frame_metrics(
            positions[index], lattices[index]
        )
        row = {
            "dataset_index": index,
            "structure_id": structure_ids[index],
            "split": split[index],
        }
        row.update(metrics)
        rows.append(row)
        for key, values in samples.items():
            samples_by_split[split[index]][key].append(values)
            all_frame_samples[key].append(values)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "training_geometry_per_structure.csv", index=False)
    per_structure_envelopes: list[dict] = []
    for _, structure in frame.iterrows():
        for prefix, metric, count in (
            ("bond", "bond_length_angstrom", 96),
            ("local_layer_spacing", "local_layer_spacing_angstrom", 64),
            ("interlayer_nn3d", "interlayer_nearest_3d_angstrom", 64),
        ):
            curve_row = {
                "dataset_index": int(structure["dataset_index"]),
                "structure_id": structure["structure_id"],
                "split": structure["split"],
                "metric": metric,
                "sample_count": count,
            }
            curve_row.update(
                {
                    name: float(structure[f"{prefix}_{name}"])
                    for name in Q_NAMES
                }
            )
            per_structure_envelopes.append(curve_row)
    pd.DataFrame(per_structure_envelopes).to_csv(
        args.output_dir / "training_geometry_per_structure_envelopes.csv", index=False
    )
    pd.DataFrame(
        {
            "dataset_index": np.arange(len(structure_ids)),
            "structure_id": structure_ids,
            "split": split,
        }
    ).to_csv(args.output_dir / "training_geometry_split_manifest.csv", index=False)

    sample_units = {
        "bond_length_angstrom": "angstrom",
        "local_layer_spacing_angstrom": "angstrom",
        "interlayer_nearest_3d_angstrom": "angstrom",
    }
    envelope_rows: list[dict] = []
    for split_name in ("train", "validation", "test"):
        for metric, unit in sample_units.items():
            values = np.concatenate(samples_by_split[split_name][metric])
            envelope_rows.append(envelope_row(metric, f"{split_name}_atomic_samples", values, unit))
        subset = frame.loc[frame["split"] == split_name]
        for metric, unit in (
            ("mean_layer_spacing_angstrom", "angstrom"),
            ("corrugation_rms_mean_angstrom", "angstrom"),
            ("corrugation_ptp_max_angstrom", "angstrom"),
            ("inplane_area_angstrom2", "angstrom^2"),
        ):
            envelope_rows.append(
                envelope_row(metric, f"{split_name}_structures", subset[metric].to_numpy(float), unit)
            )
    envelope = pd.DataFrame(envelope_rows)
    envelope.to_csv(args.output_dir / "training_geometry_envelope_summary.csv", index=False)

    train_bounds: dict[str, dict[str, float]] = {}
    for metric in sample_units:
        values = np.concatenate(samples_by_split["train"][metric])
        minimum, q005, q995, maximum = np.quantile(values, [0.0, 0.005, 0.995, 1.0])
        train_bounds[metric] = {
            "absolute_min": float(minimum),
            "robust_q005": float(q005),
            "robust_q995": float(q995),
            "absolute_max": float(maximum),
        }
    for metric in (
        "mean_layer_spacing_angstrom",
        "corrugation_rms_mean_angstrom",
        "corrugation_ptp_max_angstrom",
        "inplane_area_angstrom2",
    ):
        values = frame.loc[frame["split"] == "train", metric].to_numpy(float)
        minimum, q005, q995, maximum = np.quantile(values, [0.0, 0.005, 0.995, 1.0])
        train_bounds[metric] = {
            "absolute_min": float(minimum),
            "robust_q005": float(q005),
            "robust_q995": float(q995),
            "absolute_max": float(maximum),
        }

    coverage_rows: list[dict] = []
    for row, frame_sample_bundle in zip(rows, zip(*all_frame_samples.values())):
        coverage = {
            "dataset_index": row["dataset_index"],
            "structure_id": row["structure_id"],
            "split": row["split"],
        }
        for metric, values in zip(all_frame_samples.keys(), frame_sample_bundle):
            bounds = train_bounds[metric]
            values = np.asarray(values)
            coverage[f"{metric}_outside_robust_fraction"] = float(
                np.mean((values < bounds["robust_q005"]) | (values > bounds["robust_q995"]))
            )
            coverage[f"{metric}_outside_absolute_fraction"] = float(
                np.mean((values < bounds["absolute_min"]) | (values > bounds["absolute_max"]))
            )
        coverage_rows.append(coverage)
    pd.DataFrame(coverage_rows).to_csv(
        args.output_dir / "training_geometry_domain_coverage.csv", index=False
    )

    qa = run_geometry_qa(positions, lattices)
    if not qa["all_checks_passed"]:
        raise RuntimeError(f"Geometry QA failed: {json.dumps(qa, ensure_ascii=False)}")
    (args.output_dir / "training_geometry_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_graph": SOURCE_GRAPH,
        "source_graph_sha256": SOURCE_GRAPH_SHA256,
        "geometry_extract_sha256": sha256(args.input),
        "frozen_test_manifest": str(args.manifest),
        "frozen_test_manifest_sha256": sha256(args.manifest),
        "split": {"seed": 42, "train": 180, "validation": 60, "test": 60},
        "manifest_test_indices_verified": True,
        "manifest_test_ids_verified": True,
        "structure_count": 300,
        "atoms_per_structure": 64,
        "layer_atom_counts": [int(len(reference_lower)), int(len(reference_upper))],
        "reference_structure_id": "0_0",
        "reference_dataset_index": reference_index,
        "reference_topology": reference_topology_audit,
        "per_frame_topology": {
            "bond_count_each_frame": 96,
            "coordination_each_frame": 3,
            "adaptive_cutoff_min_angstrom": float(frame["adaptive_bond_cutoff_angstrom"].min()),
            "adaptive_cutoff_max_angstrom": float(frame["adaptive_bond_cutoff_angstrom"].max()),
            "third_neighbor_max_over_frames_angstrom": float(frame["third_neighbor_max_angstrom"].max()),
            "fourth_neighbor_min_over_frames_angstrom": float(frame["fourth_neighbor_min_angstrom"].min()),
        },
        "definitions": {
            "intralayer_bond": (
                "Three-coordinate graphene topology reconstructed independently in each "
                "frame from its clean third/fourth same-layer neighbor gap; distances use "
                "the frame-specific lattice and minimum image. This avoids assuming that "
                "atom indices preserve bond identities across independently stored frames."
            ),
            "mean_layer_spacing": "Difference of layer mean coordinates along a1 cross a2.",
            "local_layer_spacing": (
                "Per-atom perpendicular separation between the two periodic piecewise-"
                "linear layer surfaces at identical in-plane coordinates. Opposite-layer "
                "heights are obtained by periodic Delaunay interpolation and evaluated "
                "in both directions."
            ),
            "interlayer_nearest_3d": (
                "Per-atom shortest periodic three-dimensional distance to the opposite layer."
            ),
            "corrugation": "Layer height fluctuation around its layer mean along a1 cross a2.",
            "robust_envelope": "Training-set empirical 0.5th to 99.5th percentile.",
            "absolute_envelope": "Training-set empirical minimum to maximum.",
            "per_structure_envelope": (
                "Each 64-atom structure is represented by the min-max and quantile "
                "range of its 96 reconstructed bonds or 64 local perpendicular spacings; "
                "a structure mean or median is not used as the domain boundary."
            ),
        },
        "training_bounds": train_bounds,
        "geometry_qa_all_checks_passed": qa["all_checks_passed"],
        "note": (
            "Structure IDs are naturally ordered for envelope plots; this order is not "
            "claimed to be physical MD time without independent trajectory metadata."
        ),
    }
    (args.output_dir / "training_geometry_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_method_report(
        args.output_dir / "training_geometry_method_and_validation.md", metadata, qa
    )

    plot_envelopes(frame, args.figure_dir)
    plot_split_coverage(frame, args.figure_dir)
    plot_joint_domain(frame, args.figure_dir)

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "figure_dir": str(args.figure_dir),
        "training_bounds": train_bounds,
        "reference_bond_cutoff_angstrom": reference_bond_cutoff,
        "manifest_verified": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

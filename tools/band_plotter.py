#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Band Structure Plotter for OpenMX Format
================================================================================

Description:
    This script reads OpenMX band format files (openmx.Band) and generates
    publication-quality band structure figures in PNG format. It supports
    comparing multiple data sources including DFT reference, DeepH predictions,
    and ensemble member predictions.

    IMPORTANT: The .Band files from sparse_calc.jl contain eigenvalues computed
    using shift-invert method, which finds bands near Fermi level (e.g., 50
    valence + 50 conduction bands). These eigenvalues are NOT sorted by energy.

    This script implements the same band alignment strategy as Band2dat.py:
    1. Sort eigenvalues at each k-point by energy
    2. Split into valence (E <= 0) and conduction (E > 0) bands at Fermi level
    3. Align valence bands from Fermi level downward
    4. Align conduction bands from Fermi level upward
    5. Fill missing bands with NaN (matplotlib will break lines automatically)

    This ensures correct band connectivity even when different k-points have
    different numbers of bands (e.g., due to ill-projected modes).

Input Files:
    The script expects the following files in the input directory:
    - openmx_dft.Band    : DFT reference data (red scatter, smallest size)
    - openmx_mean.Band   : DeepH mean prediction (blue scatter, medium size)
    - openmx_1.Band      : Ensemble member 1 (light blue scatter, largest size)
    - openmx_2.Band      : Ensemble member 2 (light blue scatter, largest size)
    - ...
    - openmx_10.Band     : Ensemble member 10 (light blue scatter, largest size)

    Note: All files are optional. The script will plot whatever files exist.

Output:
    - PNG image file with band structure plot

Plot Style:
    All data are plotted as opaque scatter points with different sizes:
    - DFT reference (red): smallest scatter points
    - DeepH mean (blue): medium scatter points
    - Ensemble members (light blue): largest scatter points

    Figure features:
    - Thicker reference lines and grid lines
    - All four spines (top, bottom, left, right) visible with ticks
    - Larger fonts for labels and ticks
    - Adjustable figure size with adaptive font/scatter scaling

File Format (openmx.Band):
    Line 1: <num_band> 0 <fermi_level_Hartree>
    Line 2: <rlat_11> <rlat_12> ... <rlat_33> (9 values, 3x3 matrix)
    Line 3: <num_k_paths>
    Line 4~: <k_data_strings...>

    For each k-point:
        Line N:   <num_band> <kx> <ky> <kz>
        Line N+1: <E1> <E2> ... <Enum_band> (energies in Hartree)

Features:
    1. Automatic Fermi level alignment (E - E_F = 0 at Fermi surface)
    2. Greek letter display for Gamma point (Γ)
    3. Reference lines at high-symmetry k-points and Fermi level
    4. All data plotted as scatter points with size hierarchy
    5. Thicker lines and all four spines with ticks
    6. Larger fonts for better readability
    7. Adjustable aspect ratio

Usage:
    python band_plotter.py -i <input_dir> -o <output.png> [options]

Arguments:
    -i, --input_dir     Directory containing openmx_*.Band files (required)
    -o, --output        Output PNG file path (required)
    --ymin              Minimum energy in eV (default: -3.0)
    --ymax              Maximum energy in eV (default: 3.0)
    --ylim              Energy range as "ymin,ymax" (overrides --ymin/--ymax)
    --no-ensemble       Do not show ensemble bands (openmx_1-10)
    --align             Band alignment method: "fermi" or "sort" (default: fermi)
                        - fermi: split at Fermi level, align valence/conduction separately
                        - sort: simple sort by energy, n-th band = n-th eigenvalue
    --figsize           Figure size as "width,height" in inches (default: 10,8)
                        Fonts and scatter sizes scale proportionally with figure size
    --dpi               Output image DPI (default: 150)

Examples:
    # Basic usage with default energy range [-3, 3] eV
    python band_plotter.py -i ./output_dir -o band.png

    # Custom energy range
    python band_plotter.py -i ./output_dir -o band.png --ymin -2.0 --ymax 2.0

    # Using --ylim parameter
    python band_plotter.py -i ./output_dir -o band.png --ylim -2,2

    # Plot without ensemble bands
    python band_plotter.py -i ./output_dir -o band.png --no-ensemble

    # Custom figure size (width=12, height=6 inches) - fonts/scatter scale automatically
    python band_plotter.py -i ./output_dir -o band.png --figsize 12,6

Dependencies:
    - numpy
    - matplotlib
    - Python 3.6+

Author: DeepH Project
Date: 2026-03-07
================================================================================
"""

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

# Hartree to eV conversion
Hartree2eV = 1.0 / 0.036749324533634074
plt.rcParams['font.family'] = 'Nimbus Roman'


def parse_openmx_band(filename):
    """
    Parse OpenMX band format file.

    Returns:
        num_band: number of bands
        fermi_level: Fermi level in eV
        rlv: reciprocal lattice vectors (3x3)
        k_data: list of k-path strings
        kpts: k-points coordinates (nkpts, 3)
        energies: eigenvalues in eV (num_band, nkpts)
    """
    if not os.path.exists(filename):
        return None, None, None, None, None, None

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Line 1: num_band, 0, fermi_level (Hartree)
    header = lines[0].split()
    num_band = int(header[0])
    fermi_level_hartree = float(header[2])
    fermi_level = fermi_level_hartree * Hartree2eV  # Convert to eV

    # Line 2: reciprocal lattice vectors (9 values, 3x3 matrix)
    rlv = np.array([float(x) for x in lines[1].split()]).reshape(3, 3)

    # Line 3: number of k-paths
    num_paths = int(lines[2])

    # Lines 4 to 4+num_paths-1: k_data strings
    k_data = []
    for i in range(num_paths):
        k_data.append(lines[3 + i].strip())

    # Parse k-points and energies
    line_idx = 3 + num_paths
    kpts = []
    energies = []

    while line_idx < len(lines):
        line = lines[line_idx].strip()
        if not line:
            line_idx += 1
            continue

        parts = line.split()
        if len(parts) != 4:
            break

        # K-point coordinates
        kpts.append([float(x) for x in parts[1:4]])
        line_idx += 1

        # Energy values (may span multiple lines)
        e_k = []
        while line_idx < len(lines):
            next_line = lines[line_idx].strip()
            if not next_line:
                line_idx += 1
                continue

            next_parts = next_line.split()
            # Check if next k-point starts
            if len(next_parts) == 4 and next_parts[0] == str(num_band):
                break

            e_k.extend([float(x) for x in next_parts])
            line_idx += 1

        # Convert from Hartree to eV and subtract Fermi level
        e_k = np.array(e_k) * Hartree2eV - fermi_level
        energies.append(e_k)

    kpts = np.array(kpts)
    energies = np.array(energies).T  # Shape: (num_band, nkpts)

    return num_band, fermi_level, rlv, k_data, kpts, energies


def calculate_k_distances(kpts, rlv):
    """
    Calculate k-path distances in Cartesian coordinates.

    Args:
        kpts: k-points in fractional coordinates (nkpts, 3)
        rlv: reciprocal lattice vectors (3x3)

    Returns:
        distances: k-path distances (nkpts,)
    """
    kpts_cart = kpts @ rlv
    distances = np.insert(
        np.cumsum(np.linalg.norm(np.diff(kpts_cart, axis=0), axis=1)),
        0, 0.0
    )
    return distances


def extract_high_symmetry_points(k_data, distances):
    """
    Extract high symmetry point positions and labels from k_data.

    Args:
        k_data: list of k-path strings
        distances: k-path distances

    Returns:
        tick_positions: positions of high symmetry points
        tick_labels: labels for high symmetry points
    """
    tick_positions = [distances[0]]
    tick_labels = []
    current_idx = 0
    nkpts = len(distances)

    for i, path_str in enumerate(k_data):
        parts = path_str.split()
        if i == 0:
            tick_labels.append(parts[7])
        current_idx += int(parts[0])
        if current_idx <= nkpts:
            tick_positions.append(distances[current_idx - 1])
            tick_labels.append(parts[8])

    return tick_positions, tick_labels


def format_greek_label(label):
    """
    Format label with Greek letter for Gamma.
    """
    label_map = {
        'Gamma': r'$\Gamma$',
        'gamma': r'$\Gamma$',
        'GAMMA': r'$\Gamma$',
        'G': r'$\Gamma$',
        'g': r'$\Gamma$',
    }
    return label_map.get(label, label)


def align_bands_simple(energies):
    """
    Align bands using simple sort and Fermi level split.

    This is the original Band2dat.py strategy:
    1. Sort eigenvalues at each k-point by energy
    2. Split into valence (E <= 0) and conduction (E > 0) bands at Fermi level
    3. Align valence bands from Fermi level downward
    4. Align conduction bands from Fermi level upward
    5. Fill missing bands with NaN

    Args:
        energies: eigenvalues (num_band, nkpts), Fermi-aligned (E_F = 0)

    Returns:
        aligned_energies: aligned eigenvalues with NaN for missing bands
        max_valence: number of valence bands
        max_conduction: number of conduction bands
    """
    num_band, nkpts = energies.shape

    # Step 1: Sort at each k-point
    sorted_energies = np.zeros_like(energies)
    for ik in range(nkpts):
        sorted_energies[:, ik] = np.sort(energies[:, ik])

    # Step 2: Split at Fermi level (E=0) - valence and conduction
    all_valence = []
    all_conduction = []

    for ik in range(nkpts):
        e_k = sorted_energies[:, ik]
        # Split at Fermi level
        valence = e_k[e_k <= 0.0]
        conduction = e_k[e_k > 0.0]
        all_valence.append(valence)
        all_conduction.append(conduction)

    # Step 3: Find maximum number of valence and conduction bands
    max_valence = max(len(v) for v in all_valence) if all_valence else 0
    max_conduction = max(len(c) for c in all_conduction) if all_conduction else 0
    total_bands = max_valence + max_conduction

    if total_bands == 0:
        return np.full((1, nkpts), np.nan), 0, 0

    # Step 4: Build aligned energy matrix with NaN for missing bands
    aligned_energies = np.full((total_bands, nkpts), np.nan)

    for ik in range(nkpts):
        v = all_valence[ik]
        c = all_conduction[ik]

        # Valence bands: align from Fermi level downward
        if len(v) > 0:
            aligned_energies[max_valence - len(v):max_valence, ik] = v

        # Conduction bands: align from Fermi level upward
        if len(c) > 0:
            aligned_energies[max_valence:max_valence + len(c), ik] = c

    return aligned_energies, max_valence, max_conduction


def align_bands_sort_only(energies):
    """
    Align bands by simple sorting at each k-point.

    This strategy:
    1. Sort eigenvalues at each k-point from low to high
    2. Band n = n-th lowest eigenvalue at each k-point
    3. Fill missing bands with NaN

    This is the simplest alignment: the n-th band at each k-point
    corresponds to the n-th lowest eigenvalue.

    Args:
        energies: eigenvalues (num_band, nkpts), Fermi-aligned (E_F = 0)

    Returns:
        aligned_energies: aligned eigenvalues with NaN for missing bands
        max_valence: number of bands with E <= 0 at the k-point with most such bands
        max_conduction: number of bands with E > 0 at the k-point with most such bands
    """
    num_band, nkpts = energies.shape

    # Find the maximum number of bands across all k-points
    max_bands = num_band

    if max_bands == 0:
        return np.full((1, nkpts), np.nan), 0, 0

    # Build aligned energy matrix with NaN for missing bands
    aligned_energies = np.full((max_bands, nkpts), np.nan)

    for ik in range(nkpts):
        e_k = energies[:, ik]
        # Sort from low to high
        sorted_e = np.sort(e_k)
        # Fill in available bands
        aligned_energies[:len(sorted_e), ik] = sorted_e

    # Calculate valence and conduction band counts (for reference)
    max_valence = 0
    max_conduction = 0
    for ik in range(nkpts):
        e_k = aligned_energies[:, ik]
        valid_mask = ~np.isnan(e_k)
        valid_e = e_k[valid_mask]
        n_valence = np.sum(valid_e <= 0.0)
        n_conduction = np.sum(valid_e > 0.0)
        max_valence = max(max_valence, n_valence)
        max_conduction = max(max_conduction, n_conduction)

    return aligned_energies, max_valence, max_conduction


def plot_band_structure(input_dir, output_file, ylim, show_ensemble=True, align_method='fermi', figsize=None, dpi=150):
    """
    Plot band structure from multiple OpenMX band files.

    Args:
        input_dir: directory containing openmx_*.Band files
        output_file: output PNG file path
        ylim: tuple of (ymin, ymax) in eV
        show_ensemble: whether to show ensemble bands (openmx_1-10)
        align_method: band alignment method
            - 'fermi': split at Fermi level, align valence/conduction separately
            - 'sort': simple sort by energy, n-th band = n-th eigenvalue
        figsize: tuple of (width, height) in inches. If None, use default (10, 8).
        dpi: output DPI
    """
    # Select alignment function
    if align_method == 'fermi':
        align_func = align_bands_simple
    elif align_method == 'sort':
        align_func = align_bands_sort_only
    else:
        raise ValueError(f"Unknown align_method: {align_method}. Use 'fermi' or 'sort'.")

    # Parse all band files
    band_files = {}

    # DFT reference
    dft_file = os.path.join(input_dir, 'openmx_dft.Band')
    if os.path.exists(dft_file):
        band_files['dft'] = parse_openmx_band(dft_file)

    # Mean prediction
    mean_file = os.path.join(input_dir, 'openmx_mean.Band')
    if os.path.exists(mean_file):
        band_files['mean'] = parse_openmx_band(mean_file)

    # Ensemble members (0-9)
    if show_ensemble:
        for i in range(0, 9):
            ensemble_file = os.path.join(input_dir, f'openmx_{i}.Band')
            if os.path.exists(ensemble_file):
                band_files[i] = parse_openmx_band(ensemble_file)

    if not band_files:
        print("Error: No band files found!")
        return

    # Get reference data from first available file
    ref_key = 'mean' if 'mean' in band_files else list(band_files.keys())[0]
    num_band, fermi_level, rlv, k_data, kpts, energies = band_files[ref_key]

    if k_data is None:
        print("Error: Failed to parse band files!")
        return

    # Calculate k-path distances
    distances = calculate_k_distances(kpts, rlv)

    # Extract high symmetry points
    tick_positions, tick_labels = extract_high_symmetry_points(k_data, distances)

    # ========================================
    # 散点大小设置：红色最小，蓝色中等，淡蓝色最大
    # ========================================
    scatter_sizes = {
        'dft': 12,        # 红色散点，最小
        'mean': 24,      # 蓝色散点，中等
        'ensemble': 48   # 淡蓝色散点，最大
    }

    # ========================================
    # 字体大小设置（加大），根据图片尺寸自适应调整
    # ========================================
    if figsize is None:
        fig_width, fig_height = 10, 8  # 默认尺寸
    else:
        fig_width, fig_height = figsize

    # 字体大小随图片尺寸比例缩放
    scale_factor = min(fig_width / 10, fig_height / 8)
    fontsize = {
        'title': int(54 * scale_factor),
        'label': int(48 * scale_factor),
        'tick': int(48 * scale_factor),
        'legend': int(48 * scale_factor)
    }
    # 确保字体不会太小
    fontsize = {k: max(v, 10) for k, v in fontsize.items()}

    # 散点大小也随图片尺寸缩放
    scatter_sizes = {k: int(v * scale_factor) for k, v in scatter_sizes.items()}
    scatter_sizes = {k: max(v, 4) for k, v in scatter_sizes.items()}

    # ========================================
    # 参考线宽度设置（加粗），随图片尺寸缩放
    # ========================================
    ref_line_width = 3 * scale_factor
    grid_line_width = 2.5 * scale_factor
    spine_line_width = 3 * scale_factor

    # Create figure with specified size
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    # ========================================
    # Plot ensemble bands (淡蓝色散点，最大)
    # ========================================
    if show_ensemble:
        for i in range(0, 9):
            if i in band_files:
                _, _, _, _, _, ens_energies = band_files[i]
                # Align bands using selected method
                ens_aligned, n_val, n_cond = align_func(ens_energies)
                # Flatten and filter NaN for scatter plot
                ens_x = np.tile(distances, ens_aligned.shape[0])
                ens_y = ens_aligned.flatten()
                valid_mask = ~np.isnan(ens_y)
                ens_x = ens_x[valid_mask]
                ens_y = ens_y[valid_mask]
                ax.scatter(ens_x, ens_y,
                           color='lightblue', s=scatter_sizes['ensemble'],
                           alpha=1.0, zorder=1)

    # ========================================
    # Plot mean prediction (蓝色散点，中等)
    # ========================================
    if 'mean' in band_files:
        _, _, _, _, _, mean_energies = band_files['mean']
        # Align bands using selected method
        mean_aligned, n_val, n_cond = align_func(mean_energies)
        # Flatten and filter NaN for scatter plot
        mean_x = np.tile(distances, mean_aligned.shape[0])
        mean_y = mean_aligned.flatten()
        valid_mask = ~np.isnan(mean_y)
        mean_x = mean_x[valid_mask]
        mean_y = mean_y[valid_mask]
        ax.scatter(mean_x, mean_y,
                   color='blue', s=scatter_sizes['mean'],
                   alpha=1.0, zorder=2,
                   label='DeepH')

    # ========================================
    # Plot DFT reference (红色散点，最小)
    # ========================================
    if 'dft' in band_files:
        _, _, _, _, _, dft_energies = band_files['dft']
        # Align bands using selected method
        dft_aligned, n_val, n_cond = align_func(dft_energies)
        # Flatten and filter NaN for scatter plot
        dft_x = np.tile(distances, dft_aligned.shape[0])
        dft_y = dft_aligned.flatten()
        valid_mask = ~np.isnan(dft_y)
        dft_x = dft_x[valid_mask]
        dft_y = dft_y[valid_mask]
        ax.scatter(dft_x, dft_y,
                   color='red', s=scatter_sizes['dft'],
                   alpha=1.0, zorder=3,
                   label='DFT')

    # ========================================
    # Draw reference lines at high symmetry k-points (加粗)
    # ========================================
    for pos in tick_positions:
        ax.axvline(x=pos, color='black', linewidth=ref_line_width,
                   linestyle='-', zorder=0)

    # Draw Fermi level reference line (加粗)
    ax.axhline(y=0, color='black', linewidth=ref_line_width,
               linestyle='-', zorder=0)

    # ========================================
    # Set axis labels and limits (字体加大)
    # ========================================
    ax.set_ylabel(r'$E - E_F$ (eV)', fontsize=fontsize['label'])
    ax.set_xlim(distances[0], distances[-1])
    ax.set_ylim(ylim[0], ylim[1])

    # ========================================
    # Set x-axis ticks (字体加大)
    # ========================================
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([format_greek_label(label) for label in tick_labels],
                       fontsize=fontsize['tick'])

    # Set y-axis tick label font size
    ax.tick_params(axis='y', labelsize=fontsize['tick'])

    # ========================================
    # 添加右边框和上边框，并加粗所有边框
    # ========================================
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_linewidth(spine_line_width)
    ax.spines['right'].set_linewidth(spine_line_width)
    ax.spines['top'].set_linewidth(spine_line_width)
    ax.spines['bottom'].set_linewidth(spine_line_width)

    # 启用右边框和上边框的刻度
    ax.tick_params(axis='both', which='both',
                   top=True, bottom=True, left=True, right=True,
                   direction='in', length=6, width=spine_line_width)

    # ========================================
    # Add legend (横向放置于图片下方，无边框)
    # ========================================
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels,
                  handlelength=0.5,
                  loc='upper center',
                  bbox_to_anchor=(0.45, -0.05),
                  ncol=len(handles),
                  fontsize=fontsize['legend'],
                  frameon=False)

    # ========================================
    # Add grid for better readability (加粗)
    # ========================================
    ax.yaxis.grid(True, linestyle='--', alpha=0.3,
                  linewidth=grid_line_width, zorder=0)
    ax.xaxis.grid(False)  # 关闭水平网格线，用垂直参考线代替

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Band structure saved to: {output_file}")
    print(f"Energy range: [{ylim[0]:.2f}, {ylim[1]:.2f}] eV")
    print(f"High symmetry points: {[format_greek_label(l) for l in tick_labels]}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot band structure from OpenMX band files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python band_plotter.py -i ./output_dir -o band.png --ymin -3.0 --ymax 3.0
  python band_plotter.py -i ./output_dir -o band.png --ylim -2,2 --no-ensemble
  python band_plotter.py -i ./output_dir -o band.png --align sort  # Use simple sort alignment
  python band_plotter.py -i ./output_dir -o band.png --figsize 12,6  # Custom figure size
        """
    )

    parser.add_argument('-i', '--input_dir', required=True,
                        help='Directory containing openmx_*.Band files')
    parser.add_argument('-o', '--output', default=None,
                        help='Output PNG file path (default: <input_dir>/band.png)')
    parser.add_argument('--ymin', type=float, default=-3.0,
                        help='Minimum energy (eV), default: -3.0')
    parser.add_argument('--ymax', type=float, default=3.0,
                        help='Maximum energy (eV), default: 3.0')
    parser.add_argument('--ylim', type=str, default=None,
                        help='Energy range as "ymin,ymax" (eV), overrides --ymin and --ymax')
    parser.add_argument('--no-ensemble', action='store_true',
                        help='Do not show ensemble bands (openmx_1-10)')
    parser.add_argument('--align', type=str, default='fermi',
                        choices=['fermi', 'sort'],
                        help='Band alignment method: "fermi" (split at Fermi level) or "sort" (simple sort by energy). Default: fermi')
    parser.add_argument('--figsize', type=str, default=None,
                        help='Figure size as "width,height" in inches. Default: 10,8')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output DPI, default: 150')

    args = parser.parse_args()

    # Set default output path to input_dir/band.png
    output_file = args.output
    if output_file is None:
        output_file = os.path.join(args.input_dir, 'band.png')

    # Parse ylim
    if args.ylim:
        ymin, ymax = map(float, args.ylim.split(','))
        ylim = (ymin, ymax)
    else:
        ylim = (args.ymin, args.ymax)

    # Validate ylim
    if ylim[0] >= ylim[1]:
        print("Error: ymin must be less than ymax!")
        return

    # Parse figsize
    if args.figsize:
        width, height = map(float, args.figsize.split(','))
        figsize = (width, height)
    else:
        figsize = None

    # Plot band structure
    plot_band_structure(
        input_dir=args.input_dir,
        output_file=output_file,
        ylim=ylim,
        show_ensemble=not args.no_ensemble,
        align_method=args.align,
        figsize=figsize,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Sparse Hamiltonian Diagonalization for Band Structure Calculation
================================================================================

Python implementation of sparse_calc.jl for calculating band structures from
predicted Hamiltonians using sparse matrix diagonalization.

This script:
1. Reads Hamiltonian and overlap matrices from HDF5 files
2. Constructs sparse matrices in real space H(R) and S(R)
3. Performs Bloch sum to get k-space matrices H(k) and S(k)
4. Solves generalized eigenvalue problem H(k)|ψ⟩ = E·S(k)|ψ⟩
5. Outputs band structure in OpenMX format

Usage:
    python sparse_calc.py -i <input_dir> -o <output_dir> --config <config.json>

Dependencies:
    - numpy, scipy, h5py
================================================================================
"""

import argparse
import json
import os
import pickle
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
from scipy import linalg
from scipy.sparse import csc_matrix, csr_matrix, hstack, vstack
from scipy.sparse.linalg import eigs, LinearOperator, splu


# Constants
EV2HARTREE = 0.036749324533634074
BOHR2ANG = 0.529177249
DEFAULT_DTYPE = np.complex128


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate band structure from predicted Hamiltonians"
    )
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        default="./",
        help="Path containing rlat.dat, orbital_types.dat, site_positions.dat, "
             "hamiltonians_pred.h5, and overlaps.h5"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./",
        help="Path for output openmx.Band"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config file in JSON format"
    )
    parser.add_argument(
        "--ill_project",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help="Project out ill-conditioned eigenvectors of overlap matrix"
    )
    parser.add_argument(
        "--ill_threshold",
        type=float,
        default=5e-4,
        help="Threshold for ill_project"
    )
    return parser.parse_args()


def read_h5_to_dict(filename):
    """
    Read HDF5 file and return a dictionary with integer keys.

    The key format in HDF5 is like "[R1, R2, R3, i, j]",
    which is converted to tuple (R1, R2, R3, i, j).
    """
    result = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            # Parse key: "[R1, R2, R3, i, j]" -> (R1, R2, R3, i, j)
            key_str = key[1:-1]  # Remove brackets
            indices = tuple(map(int, key_str.split(',')))
            # Transpose to match Julia's permutedims
            result[indices] = f[key][()].T
    return result


def k_data_to_num_ks(kdata):
    """Extract number of k-points from k_data string."""
    return int(kdata.split()[0])


def k_data_to_kpath(kdata):
    """Extract k-path from k_data string."""
    return list(map(float, kdata.split()[1:7]))


def genlist(x):
    """Generate linearly spaced array."""
    return np.linspace(x[0], x[1], int(x[2]))


def std_out_array(a):
    """Convert array to space-separated string."""
    return ' '.join(map(str, np.asarray(a).flatten()))


def construct_mesh_kpts(nkmesh, offset=None, k1=None, k2=None):
    """Construct k-point mesh for DOS calculation."""
    if offset is None:
        offset = [0.0, 0.0, 0.0]
    if k1 is None:
        k1 = [0.0, 0.0, 0.0]
    if k2 is None:
        k2 = [1.0, 1.0, 1.0]

    nkmesh = list(nkmesh)
    nkpts = nkmesh[0] * nkmesh[1] * nkmesh[2]
    kpts = np.zeros((3, nkpts))

    idx = 0
    for ikx in range(nkmesh[0]):
        for iky in range(nkmesh[1]):
            for ikz in range(nkmesh[2]):
                kpts[0, idx] = ikx / nkmesh[0] * (k2[0] - k1[0]) + k1[0]
                kpts[1, idx] = iky / nkmesh[1] * (k2[1] - k1[1]) + k1[1]
                kpts[2, idx] = ikz / nkmesh[2] * (k2[2] - k1[2]) + k1[2]
                idx += 1

    return kpts + np.array(offset).reshape(3, 1)


class ShiftInvertLinearOperator:
    """
    Linear operator for shift-invert eigenvalue solving.

    Implements (H - sigma * S)^(-1) @ S for finding eigenvalues near sigma.
    Uses sparse LU decomposition for efficient solving.
    """

    def __init__(self, H, S, sigma):
        """
        Initialize the linear operator.

        Args:
            H: Hamiltonian matrix (sparse or dense)
            S: Overlap matrix (sparse or dense)
            sigma: Shift value (typically fermi_level)
        """
        self.n = H.shape[0]
        self.sigma = sigma

        # Compute H - sigma * S
        if isinstance(H, csc_matrix):
            A = H - sigma * S
            # Make sure A is Hermitian for LU decomposition
            A = (A + A.conj().T) / 2
            self.A = A.tocsc()
            self.use_sparse = True
            # Perform LU decomposition
            self.lu = splu(self.A)
        else:
            self.A = (H + H.conj().T) / 2 - sigma * (S + S.conj().T) / 2
            self.use_sparse = False

        # Store S for the matvec operation
        if isinstance(S, csc_matrix):
            self.S = S
            self.S_sparse = True
        else:
            self.S = S
            self.S_sparse = False

        # Create LinearOperator
        self.linop = LinearOperator(
            shape=(self.n, self.n),
            matvec=self._matvec,
            dtype=np.complex128
        )

    def _matvec(self, x):
        """Compute (H - sigma * S)^(-1) @ S @ x."""
        # First multiply by S
        if self.S_sparse:
            Sx = self.S @ x
        else:
            Sx = np.dot(self.S, x)

        # Then solve (H - sigma * S)^(-1) @ Sx
        if self.use_sparse:
            return self.lu.solve(Sx)
        else:
            return linalg.solve(self.A, Sx)

    def __matmul__(self, x):
        return self._matvec(x)


def solve_eigenvalue_shift_invert(H_k, S_k, fermi_level, num_band, max_iter,
                                   ill_project=True, ill_threshold=5e-4,
                                   return_eigenvectors=True):
    """
    Solve generalized eigenvalue problem using shift-invert method.

    Finds eigenvalues near fermi_level by solving:
    (H - Ef * S)^(-1) @ S @ |ψ⟩ = (1/(E - Ef)) @ |ψ⟩

    Args:
        H_k: Hamiltonian at k-point
        S_k: Overlap matrix at k-point
        fermi_level: Fermi energy (shift value)
        num_band: Number of bands to compute
        max_iter: Maximum iterations for eigensolver
        ill_project: Whether to project out ill-conditioned eigenvectors
        ill_threshold: Threshold for ill-conditioned eigenvalues
        return_eigenvectors: Whether to return eigenvectors

    Returns:
        Eigenvalues (and optionally eigenvectors)
    """
    n = H_k.shape[0]

    # Ensure Hermitian
    H_k = (H_k + H_k.conj().T) / 2
    S_k = (S_k + S_k.conj().T) / 2

    # Create linear operator for shift-invert
    shift_inv = ShiftInvertLinearOperator(H_k, S_k, fermi_level)

    try:
        if return_eigenvectors:
            # Use ARPACK via scipy.sparse.linalg.eigs
            egval_inv, egvec_sub = eigs(
                shift_inv.linop,
                k=num_band,
                which='LM',  # Largest magnitude (closest to fermi_level after inversion)
                maxiter=max_iter,
                v0=np.random.rand(n) + 1j * np.random.rand(n)
            )

            # Convert back to actual eigenvalues
            egval = np.real(1.0 / egval_inv) + fermi_level

            if ill_project:
                # Orthogonalize eigenvectors using QR decomposition
                Q, R = np.linalg.qr(egvec_sub)
                egvec_sub = Q.astype(np.complex128)

                # Project onto well-conditioned subspace
                S_sub = egvec_sub.conj().T @ S_k @ egvec_sub
                egval_S, egvec_S = linalg.eigh(S_sub)

                # Find well-conditioned indices
                project_index = np.abs(egval_S) > ill_threshold
                n_ill = np.sum(~project_index)

                if n_ill > 0:
                    warnings.warn(
                        f"ill-conditioned eigenvalues detected, projected out {n_ill} eigenvalues"
                    )
                    # Project out ill-conditioned modes
                    egvec_S_good = egvec_S[:, project_index]

                    H_sub = egvec_sub.conj().T @ H_k @ egvec_sub
                    H_sub_proj = egvec_S_good.conj().T @ H_sub @ egvec_S_good
                    S_sub_proj = egvec_S_good.conj().T @ S_sub @ egvec_S_good

                    egval_good, egvec_good = linalg.eigh(H_sub_proj, S_sub_proj)

                    # Pad with large values for ill-conditioned modes
                    egval = np.concatenate([egval_good, np.full(n_ill, 1e4)])
                    egvec = egvec_sub @ egvec_S_good @ egvec_good

                    return egval, egvec
                else:
                    return egval, egvec_sub
            else:
                return egval, egvec_sub
        else:
            egval_inv = eigs(
                shift_inv.linop,
                k=num_band,
                which='LM',
                maxiter=max_iter,
                return_eigenvectors=False,
                v0=np.random.rand(n) + 1j * np.random.rand(n)
            )
            egval = np.real(1.0 / egval_inv) + fermi_level
            return egval

    except Exception as e:
        warnings.warn(f"eigs failed, falling back to dense solver: {e}")
        # Fallback to dense solver
        if isinstance(H_k, csc_matrix):
            H_k = H_k.toarray()
            S_k = S_k.toarray()

        egval, egvec = linalg.eigh(H_k, S_k)
        egval = egval[:num_band]

        if return_eigenvectors:
            return egval, egvec[:, :num_band]
        return egval


def construct_k_space_matrix(H_R, S_R, k, norbits):
    """
    Construct k-space Hamiltonian and overlap matrices via Bloch sum.

    H(k) = Σ_R H(R) * exp(i * 2π * k · R)
    S(k) = Σ_R S(R) * exp(i * 2π * k · R)
    """
    H_k = csc_matrix((norbits, norbits), dtype=DEFAULT_DTYPE)
    S_k = csc_matrix((norbits, norbits), dtype=DEFAULT_DTYPE)

    for R, H_R_block in H_R.items():
        phase = np.exp(2j * np.pi * np.dot(k, R))
        H_k = H_k + H_R_block * phase
        S_R_block = S_R[R]
        S_k = S_k + S_R_block * phase

    # Ensure Hermitian
    H_k = (H_k + H_k.conj().T) / 2
    S_k = (S_k + S_k.conj().T) / 2

    return H_k, S_k


def main():
    args = parse_args()

    print(f"Config file: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)

    calc_job = config["calc_job"]
    ill_project = args.ill_project
    ill_threshold = args.ill_threshold

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check spinful
    info_file = input_dir / "info.json"
    if info_file.exists():
        with open(info_file, 'r') as f:
            info = json.load(f)
        spinful = info.get("isspinful", False)
    else:
        spinful = False

    # Read site positions
    site_positions = np.loadtxt(input_dir / "site_positions.dat")
    nsites = site_positions.shape[1]

    # Read orbital types
    with open(input_dir / "orbital_types.dat", 'r') as f:
        orbital_types = [list(map(int, line.split())) for line in f]

    # Calculate number of orbitals per site
    # Each orbital type l has (2l+1) orbitals
    site_norbits = np.array([sum((l * 2 + 1) for l in ot) for ot in orbital_types])
    if spinful:
        site_norbits *= 2
    norbits = int(np.sum(site_norbits))
    site_norbits_cumsum = np.cumsum(site_norbits)

    # Read reciprocal lattice
    rlat = np.loadtxt(input_dir / "rlat.dat")

    # Load or construct sparse matrices
    sparse_cache_file = input_dir / "sparse_matrix.pkl"

    if sparse_cache_file.exists():
        print(f"Reading sparse matrix from {sparse_cache_file}")
        with open(sparse_cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        H_R = cache_data['H_R']
        S_R = cache_data['S_R']
    else:
        print("Reading h5 files...")
        begin_time = time.time()

        hamiltonians_pred = read_h5_to_dict(input_dir / "hamiltonians_pred.h5")
        overlaps = read_h5_to_dict(input_dir / "overlaps.h5")
        print(f"Time for reading h5: {time.time() - begin_time:.2f}s")

        print("Constructing sparse matrices...")
        begin_time = time.time()

        # Convert R keys from list to tuple for consistency
        H_R = {}
        S_R = {}

        for key in hamiltonians_pred:
            R = tuple(key[0:3])
            atom_i = key[3] - 1  # Convert to 0-indexed
            atom_j = key[4] - 1

            H_block = hamiltonians_pred[key]

            # Get overlap matrix
            if key in overlaps:
                S_block = overlaps[key]
                if spinful:
                    # Expand overlap for spinful case
                    n = S_block.shape[0]
                    S_block = np.block([
                        [S_block, np.zeros((n, n))],
                        [np.zeros((n, n)), S_block]
                    ])
            else:
                S_block = np.zeros_like(H_block)

            # Build sparse matrix for this R
            ni = int(site_norbits[atom_i])
            nj = int(site_norbits[atom_j])

            i_start = int(site_norbits_cumsum[atom_i] - site_norbits[atom_i])
            j_start = int(site_norbits_cumsum[atom_j] - site_norbits[atom_j])

            # Create index arrays
            rows = []
            cols = []
            H_vals = []
            S_vals = []

            for bi in range(ni):
                for bj in range(nj):
                    rows.append(i_start + bi)
                    cols.append(j_start + bj)
                    H_vals.append(H_block[bi, bj])
                    S_vals.append(S_block[bi, bj])

            # Convert R to tuple for dictionary key
            if R not in H_R:
                H_R[R] = csc_matrix((norbits, norbits), dtype=DEFAULT_DTYPE)
                S_R[R] = csc_matrix((norbits, norbits), dtype=DEFAULT_DTYPE)

            H_R[R] = H_R[R] + csc_matrix((H_vals, (rows, cols)), shape=(norbits, norbits))
            S_R[R] = S_R[R] + csc_matrix((S_vals, (rows, cols)), shape=(norbits, norbits))

        print(f"Time for constructing sparse matrices: {time.time() - begin_time:.2f}s")

        # Cache the matrices
        with open(sparse_cache_file, 'wb') as f:
            pickle.dump({'H_R': H_R, 'S_R': S_R}, f)

    # Band calculation
    if calc_job == "band":
        which_k = config.get("which_k", 0)
        fermi_level = config["fermi_level"]
        max_iter = config["max_iter"]
        num_band = config["num_band"]
        k_data = config["k_data"]

        print("Calculating bands...")
        num_ks = [k_data_to_num_ks(kd) for kd in k_data]
        kpaths = [k_data_to_kpath(kd) for kd in k_data]

        total_ks = sum(num_ks)
        egvals = np.zeros((num_band, total_ks))

        begin_time = time.time()
        idx_k = 1

        for i, kpath in enumerate(kpaths):
            pnkpts = num_ks[i]
            kxs = np.linspace(kpath[0], kpath[3], pnkpts)
            kys = np.linspace(kpath[1], kpath[4], pnkpts)
            kzs = np.linspace(kpath[2], kpath[5], pnkpts)

            for kx, ky, kz in zip(kxs, kys, kzs):
                if which_k == 0 or which_k == idx_k:
                    k = np.array([kx, ky, kz])

                    # Construct k-space matrices
                    H_k, S_k = construct_k_space_matrix(H_R, S_R, k, norbits)

                    print(f"Time for No.{idx_k} matrix construction: {time.time() - begin_time:.2f}s")

                    # Solve eigenvalue problem
                    egval = solve_eigenvalue_shift_invert(
                        H_k, S_k, fermi_level, num_band, max_iter,
                        ill_project=ill_project,
                        ill_threshold=ill_threshold,
                        return_eigenvectors=False
                    )

                    egvals[:, idx_k - 1] = egval

                    if which_k != 0:
                        # Save single k-point results
                        np.savetxt(output_dir / "kpoint.dat", [kx, ky, kz])
                        np.savetxt(output_dir / "egval.dat", egval)

                    print(f"Time for solving No.{idx_k} eigenvalues at k = [{kx:.4f}, {ky:.4f}, {kz:.4f}]: {time.time() - begin_time:.2f}s")

                idx_k += 1

        # Output in OpenMX band format
        with open(output_dir / "openmx.Band", 'w') as f:
            f.write(f"{num_band} 0 {EV2HARTREE * fermi_level}\n")

            openmx_rlat = (rlat * BOHR2ANG).flatten()
            f.write(f"{std_out_array(openmx_rlat)}\n")

            f.write(f"{len(k_data)}\n")
            for line in k_data:
                f.write(f"{line}\n")

            idx_k = 1
            for i, kpath in enumerate(kpaths):
                pnkpts = num_ks[i]
                kstart = kpath[0:3]
                kend = kpath[3:6]

                for j in range(pnkpts):
                    alpha = j / (pnkpts - 1) if pnkpts > 1 else 0
                    kvec = [kstart[0] + alpha * (kend[0] - kstart[0]),
                            kstart[1] + alpha * (kend[1] - kstart[1]),
                            kstart[2] + alpha * (kend[2] - kstart[2])]

                    f.write(f"{num_band} {std_out_array(kvec)}\n")
                    f.write(f"{std_out_array(EV2HARTREE * egvals[:, idx_k - 1])}\n")
                    idx_k += 1

        print(f"Band structure saved to {output_dir / 'openmx.Band'}")

    # DOS calculation
    elif calc_job == "dos":
        fermi_level = config["fermi_level"]
        max_iter = config["max_iter"]
        num_band = config["num_band"]
        nkmesh = config["kmesh"]

        print("Calculating DOS...")
        ks = construct_mesh_kpts(nkmesh)
        nks = ks.shape[1]

        egvals = np.zeros((num_band, nks))
        begin_time = time.time()

        for idx_k in range(nks):
            kx, ky, kz = ks[:, idx_k]
            k = np.array([kx, ky, kz])

            # Construct k-space matrices
            H_k, S_k = construct_k_space_matrix(H_R, S_R, k, norbits)

            print(f"Time for No.{idx_k + 1} matrix construction: {time.time() - begin_time:.2f}s")

            # Solve eigenvalue problem
            egval = solve_eigenvalue_shift_invert(
                H_k, S_k, fermi_level, num_band, max_iter,
                ill_project=ill_project,
                ill_threshold=ill_threshold,
                return_eigenvectors=False
            )

            egvals[:, idx_k] = egval
            print(f"Time for solving No.{idx_k + 1} eigenvalues at k = [{kx:.4f}, {ky:.4f}, {kz:.4f}]: {time.time() - begin_time:.2f}s")

        np.savetxt(output_dir / "egvals.dat", egvals)

        # Calculate DOS with Gaussian broadening
        epsilon = config["epsilon"]
        omegas = genlist(config["omegas"])
        n_omegas = len(omegas)
        dos = np.zeros(n_omegas)

        factor = 1.0 / ((2 * np.pi) ** 3 * epsilon * np.sqrt(np.pi))

        for idx_k in range(nks):
            for idx_band in range(num_band):
                for idx_omega, omega in enumerate(omegas):
                    dos[idx_omega] += np.exp(
                        -((egvals[idx_band, idx_k] - omega - fermi_level) ** 2) / (epsilon ** 2)
                    ) * factor

        np.savetxt(output_dir / "dos.dat", np.column_stack([omegas, dos]))
        print(f"DOS saved to {output_dir / 'dos.dat'}")


if __name__ == "__main__":
    main()

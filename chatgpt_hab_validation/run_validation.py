#!/usr/bin/env python3
"""Validate low-rank A-B coupling and dressed-atom ideas for H/F/Cl/Br/I diatomics.

Primary protocol:
- restricted Hartree-Fock (RHF)
- PySCF MINAO basis
- 1D bond-length scan for each of 15 unordered pairs
- meta-Lowdin orthogonal AO representation
- compare Fock-block rank, occupied-projector rank, local natural occupations,
  joint cross-partner port stability, and a two-state projected effective Hamiltonian.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from pyscf import gto, lo, scf

HARTREE_TO_EV = 27.211386245988
ELEMENTS = ("H", "F", "Cl", "Br", "I")
VALENCE_N = {"H": 1, "F": 2, "Cl": 3, "Br": 4, "I": 5}
COVALENT_RADIUS_ANG = {"H": 0.31, "F": 0.64, "Cl": 0.99, "Br": 1.14, "I": 1.33}
RANK_REL_TOLS = (1e-1, 1e-2, 1e-3, 1e-4)
ETA = 0.05


def jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        if np.iscomplexobj(x):
            return {"real": x.real.tolist(), "imag": x.imag.tolist()}
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, complex):
        return {"real": float(x.real), "imag": float(x.imag)}
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def invsqrt_sym(a: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    w, u = np.linalg.eigh((a + a.T.conj()) * 0.5)
    if np.min(w) < tol:
        raise np.linalg.LinAlgError(f"Matrix not positive definite; min eigenvalue={np.min(w):.3e}")
    return (u * (w ** -0.5)) @ u.T.conj()


def sv_metrics(mat: np.ndarray) -> Dict[str, Any]:
    s = np.asarray(np.linalg.svd(mat, compute_uv=False), dtype=float)
    if s.size == 0 or float(np.sum(s * s)) == 0.0:
        return {
            "singular_values": s,
            "rank_abs_1e-10": 0,
            **{f"rank_rel_{tol:g}": 0 for tol in RANK_REL_TOLS},
            "first_sq_fraction": 0.0,
            "first2_sq_fraction": 0.0,
            "stable_rank": 0.0,
            "entropy_rank": 0.0,
        }
    smax = float(s[0])
    sq = s * s
    total = float(np.sum(sq))
    p = sq / total
    entropy = -float(np.sum(p[p > 0] * np.log(p[p > 0])))
    return {
        "singular_values": s,
        "rank_abs_1e-10": int(np.sum(s > 1e-10)),
        **{f"rank_rel_{tol:g}": int(np.sum(s > tol * smax)) for tol in RANK_REL_TOLS},
        "first_sq_fraction": float(sq[0] / total),
        "first2_sq_fraction": float(np.sum(sq[:2]) / total),
        "stable_rank": float(total / (smax * smax)),
        "entropy_rank": float(math.exp(entropy)),
    }


def make_mol(a: str, b: str, r_ang: float, basis: str = "minao") -> gto.Mole:
    return gto.M(
        atom=[(a, (0.0, 0.0, -0.5 * r_ang)), (b, (0.0, 0.0, 0.5 * r_ang))],
        unit="Angstrom",
        basis=basis,
        charge=0,
        spin=0,
        symmetry=False,
        cart=False,
        verbose=0,
    )


def run_rhf(mol: gto.Mole) -> scf.hf.RHF:
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-7
    mf.max_cycle = 160
    mf.diis_space = 12
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-7
        mf.max_cycle = 80
        mf.verbose = 0
        mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge")
    return mf


def scan_equilibrium(a: str, b: str, basis: str = "minao") -> Tuple[float, float, List[Tuple[float, float]]]:
    r0 = COVALENT_RADIUS_ANG[a] + COVALENT_RADIUS_ANG[b]
    coarse = np.linspace(0.75 * r0, 1.38 * r0, 13)
    evaluated: Dict[float, float] = {}

    def eval_r(r: float) -> float:
        key = round(float(r), 8)
        if key not in evaluated:
            mf = run_rhf(make_mol(a, b, float(r), basis))
            evaluated[key] = float(mf.e_tot)
        return evaluated[key]

    for r in coarse:
        eval_r(float(r))
    rs = np.array(sorted(evaluated))
    es = np.array([evaluated[float(r)] for r in rs])
    i = int(np.argmin(es))
    step = float(coarse[1] - coarse[0])
    center = float(rs[i])
    fine = np.linspace(max(0.45, center - 1.25 * step), center + 1.25 * step, 13)
    for r in fine:
        eval_r(float(r))

    rs = np.array(sorted(evaluated))
    es = np.array([evaluated[float(r)] for r in rs])
    i = int(np.argmin(es))
    r_best = float(rs[i])
    e_best = float(es[i])
    if 0 < i < len(rs) - 1:
        x = rs[i - 1 : i + 2]
        y = es[i - 1 : i + 2]
        coeff = np.polyfit(x, y, 2)
        if coeff[0] > 0:
            r_quad = float(-coeff[1] / (2 * coeff[0]))
            if float(x[0]) <= r_quad <= float(x[-1]):
                e_quad = eval_r(r_quad)
                if e_quad < e_best:
                    r_best, e_best = r_quad, e_quad
    curve = sorted((float(r), float(e)) for r, e in evaluated.items())
    return r_best, e_best, curve


def meta_lowdin_coeff(mol: gto.Mole, s: np.ndarray) -> Tuple[np.ndarray, str]:
    try:
        c = lo.orth.orth_ao(mol, method="meta_lowdin", pre_orth_ao="ANO", s=s)
        method = "meta_lowdin/ANO"
    except Exception:
        c = invsqrt_sym(s)
        method = "lowdin_fallback"
    err = np.linalg.norm(c.T @ s @ c - np.eye(s.shape[0]))
    if err > 1e-7:
        raise RuntimeError(f"AO orthogonalization failed: ||C^TSC-I||={err:.3e}")
    return c, method


def valence_indices(mol: gto.Mole, atom_id: int, element: str) -> List[int]:
    labels = mol.ao_labels()
    n = VALENCE_N[element]
    p0, p1 = mol.aoslice_by_atom()[atom_id][2:4]
    out: List[int] = []
    for i in range(int(p0), int(p1)):
        parts = labels[i].split()
        tok = parts[2] if len(parts) >= 3 else ""
        if tok.startswith(f"{n}s") or tok.startswith(f"{n}p"):
            out.append(i)
    if not out:
        raise RuntimeError(f"No valence AOs found for atom {atom_id} {element}; labels={labels[p0:p1]}")
    return out


def endpoint_payload(mol, fock_o, hcore_o, dens_o, atom_id, element, partner, homo, lumo):
    slices = mol.aoslice_by_atom()
    a0, a1 = map(int, slices[atom_id][2:4])
    other_id = 1 - atom_id
    b0, b1 = map(int, slices[other_id][2:4])
    ia = list(range(a0, a1))
    ib = list(range(b0, b1))
    va = valence_indices(mol, atom_id, element)
    vb = valence_indices(mol, other_id, partner)

    f_ab_all = fock_o[np.ix_(ia, ib)]
    f_ab_val = fock_o[np.ix_(va, vb)]
    h_ab_val = hcore_o[np.ix_(va, vb)]
    p_ab_all = dens_o[np.ix_(ia, ib)]
    p_ab_val = dens_o[np.ix_(va, vb)]
    p_aa_all = dens_o[np.ix_(ia, ia)]
    p_aa_val = dens_o[np.ix_(va, va)]
    f_aa_val = fock_o[np.ix_(va, va)]

    id_lhs = p_ab_all @ p_ab_all.T
    id_rhs = 2.0 * p_aa_all - p_aa_all @ p_aa_all
    projector_identity_relerr = float(np.linalg.norm(id_lhs - id_rhs) / max(np.linalg.norm(id_rhs), 1e-14))

    occ, u_occ = np.linalg.eigh((p_aa_val + p_aa_val.T) * 0.5)
    order = np.argsort(occ)[::-1]
    occ = occ[order]
    u_occ = u_occ[:, order]
    activity = occ * (2.0 - occ)
    active_idx = int(np.argmax(activity))
    active_vec = u_occ[:, active_idx]

    uf, sf, vhf = np.linalg.svd(f_ab_val, full_matrices=False)
    up, sp, vhp = np.linalg.svd(p_ab_val, full_matrices=False)
    dominant_fock_vec = uf[:, 0] if uf.shape[1] else np.zeros(len(va))
    dominant_density_vec = up[:, 0] if up.shape[1] else np.zeros(len(va))

    z_values = {"vacuum": 0.0 + 1j * ETA, "midgap": 0.5 * (homo + lumo) + 1j * ETA}
    self_energies: Dict[str, Any] = {}
    for name, z in z_values.items():
        f_bb = fock_o[np.ix_(ib, ib)]
        f_a_b = fock_o[np.ix_(va, ib)]
        sigma = f_a_b @ np.linalg.solve(z * np.eye(len(ib)) - f_bb, f_a_b.T)
        k_eff = f_aa_val + sigma
        self_energies[name] = {
            "z": z,
            "sigma": sigma,
            "k_eff": k_eff,
            "sigma_fro": float(np.linalg.norm(sigma)),
            "k_eff_eigs_real": np.linalg.eigvalsh((k_eff.real + k_eff.real.T) * 0.5),
        }

    return {
        "atom_id": atom_id,
        "element": element,
        "partner": partner,
        "all_indices": ia,
        "partner_all_indices": ib,
        "valence_indices": va,
        "partner_valence_indices": vb,
        "ao_labels_valence": [mol.ao_labels()[i] for i in va],
        "fock_cross_all": f_ab_all,
        "fock_cross_valence": f_ab_val,
        "hcore_cross_valence": h_ab_val,
        "density_cross_all": p_ab_all,
        "density_cross_valence": p_ab_val,
        "fock_cross_all_metrics": sv_metrics(f_ab_all),
        "fock_cross_valence_metrics": sv_metrics(f_ab_val),
        "hcore_cross_valence_metrics": sv_metrics(h_ab_val),
        "density_cross_all_metrics": sv_metrics(p_ab_all),
        "density_cross_valence_metrics": sv_metrics(p_ab_val),
        "projector_identity_relerr": projector_identity_relerr,
        "local_density_valence": p_aa_val,
        "local_density_eigenvalues": occ,
        "local_density_activity": activity,
        "active_local_index": active_idx,
        "active_local_vector": active_vec,
        "dominant_fock_port": dominant_fock_vec,
        "dominant_density_port": dominant_density_vec,
        "active_fock_port_overlap": float(abs(np.vdot(active_vec, dominant_fock_vec))),
        "active_density_port_overlap": float(abs(np.vdot(active_vec, dominant_density_vec))),
        "local_fock_valence": f_aa_val,
        "local_fock_eigenvalues": np.linalg.eigvalsh((f_aa_val + f_aa_val.T) * 0.5),
        "self_energies": self_energies,
    }


def two_state_effective_hamiltonian(fock_o, mo_occ, endpoint_a, endpoint_b):
    n = fock_o.shape[0]
    va = endpoint_a["valence_indices"]
    vb = endpoint_b["valence_indices"]
    ua = np.asarray(endpoint_a["active_local_vector"])
    ub = np.asarray(endpoint_b["active_local_vector"])
    pcols = np.zeros((n, 2))
    pcols[va, 0] = ua
    pcols[vb, 1] = ub
    pcols = pcols @ invsqrt_sym(pcols.T @ pcols)

    evals, evecs = np.linalg.eigh((fock_o + fock_o.T) * 0.5)
    weights = np.sum(np.abs(pcols.T @ evecs) ** 2, axis=0)
    occ_idx = np.where(np.asarray(mo_occ) > 0)[0]
    vir_idx = np.where(np.asarray(mo_occ) == 0)[0]
    i_occ = int(occ_idx[np.argmax(weights[occ_idx])])
    i_vir = int(vir_idx[np.argmax(weights[vir_idx])])
    selected = [i_occ, i_vir]
    psi = evecs[:, selected]
    bmat = pcols.T @ psi
    sproj = bmat.T @ bmat
    capture = float(np.trace(sproj) / 2.0)
    cond = float(np.linalg.cond(bmat))
    phi = bmat @ invsqrt_sym(sproj)
    heff = phi @ np.diag(evals[selected]) @ phi.T
    return {
        "selected_state_indices": selected,
        "selected_state_energies": evals[selected],
        "selected_state_weights": weights[selected],
        "projection_capture": capture,
        "projection_condition": cond,
        "heff_2x2": heff,
        "epsilon_a": float(heff[0, 0]),
        "epsilon_b": float(heff[1, 1]),
        "t_ab": float(heff[0, 1]),
    }


def analyze_pair(a: str, b: str, r_ang: float, basis: str = "minao") -> Dict[str, Any]:
    mol = make_mol(a, b, r_ang, basis)
    mf = run_rhf(mol)
    s = mol.intor_symmetric("int1e_ovlp")
    fock = mf.get_fock()
    hcore = mf.get_hcore()
    c_orth, orth_method = meta_lowdin_coeff(mol, s)
    fock_o = c_orth.T @ fock @ c_orth
    hcore_o = c_orth.T @ hcore @ c_orth
    cmo_o = np.linalg.solve(c_orth, mf.mo_coeff)
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    dens_o = 2.0 * cmo_o[:, :nocc] @ cmo_o[:, :nocc].T
    idempotency_relerr = float(np.linalg.norm(dens_o @ dens_o - 2.0 * dens_o) / max(np.linalg.norm(2.0 * dens_o), 1e-14))
    homo = float(mf.mo_energy[nocc - 1])
    lumo = float(mf.mo_energy[nocc])
    ep0 = endpoint_payload(mol, fock_o, hcore_o, dens_o, 0, a, b, homo, lumo)
    ep1 = endpoint_payload(mol, fock_o, hcore_o, dens_o, 1, b, a, homo, lumo)
    two = two_state_effective_hamiltonian(fock_o, mf.mo_occ, ep0, ep1)
    return {
        "pair": f"{a}-{b}", "a": a, "b": b, "basis": basis,
        "r_ang": float(r_ang), "e_tot": float(mf.e_tot),
        "nao": int(mol.nao_nr()), "nelectron": int(mol.nelectron),
        "nocc": nocc, "nvirt": int(mol.nao_nr() - nocc),
        "orthogonalization": orth_method,
        "orth_error": float(np.linalg.norm(c_orth.T @ s @ c_orth - np.eye(mol.nao_nr()))),
        "density_idempotency_relerr": idempotency_relerr,
        "homo": homo, "lumo": lumo, "gap": lumo - homo,
        "ao_labels": mol.ao_labels(), "fock_orth": fock_o, "density_orth": dens_o,
        "endpoints": [ep0, ep1], "two_state": two,
    }


def endpoint_rows(results: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for res in results:
        two = res["two_state"]
        for i, ep in enumerate(res["endpoints"]):
            fm = ep["fock_cross_valence_metrics"]
            pm = ep["density_cross_valence_metrics"]
            hm = ep["hcore_cross_valence_metrics"]
            occ = np.asarray(ep["local_density_eigenvalues"])
            activity = np.asarray(ep["local_density_activity"])
            row: Dict[str, Any] = {
                "pair": res["pair"], "element": ep["element"], "partner": ep["partner"],
                "r_ang": res["r_ang"], "nao": res["nao"], "nvirt": res["nvirt"],
                "valence_dim": len(ep["valence_indices"]),
                "partner_valence_dim": len(ep["partner_valence_indices"]),
                "fock_rank_rel_1e-2": fm["rank_rel_0.01"],
                "fock_rank_rel_1e-3": fm["rank_rel_0.001"],
                "fock_first_sq_fraction": fm["first_sq_fraction"],
                "fock_stable_rank": fm["stable_rank"],
                "hcore_rank_rel_1e-2": hm["rank_rel_0.01"],
                "hcore_first_sq_fraction": hm["first_sq_fraction"],
                "density_rank_rel_1e-2": pm["rank_rel_0.01"],
                "density_rank_rel_1e-3": pm["rank_rel_0.001"],
                "density_first_sq_fraction": pm["first_sq_fraction"],
                "density_stable_rank": pm["stable_rank"],
                "projector_identity_relerr": ep["projector_identity_relerr"],
                "active_fock_port_overlap": ep["active_fock_port_overlap"],
                "active_density_port_overlap": ep["active_density_port_overlap"],
                "active_local_occupation": float(occ[int(ep["active_local_index"])]),
                "max_local_activity": float(np.max(activity)),
                "sigma_vacuum_fro": ep["self_energies"]["vacuum"]["sigma_fro"],
                "sigma_midgap_fro": ep["self_energies"]["midgap"]["sigma_fro"],
                "two_state_projection_capture": two["projection_capture"],
                "two_state_projection_condition": two["projection_condition"],
                "two_state_epsilon_endpoint": two["epsilon_a"] if i == 0 else two["epsilon_b"],
                "two_state_epsilon_partner": two["epsilon_b"] if i == 0 else two["epsilon_a"],
                "two_state_t_abs": abs(two["t_ab"]),
            }
            for j, val in enumerate(occ): row[f"local_occ_{j+1}"] = float(val)
            for j, val in enumerate(fm["singular_values"]): row[f"fock_sv_{j+1}"] = float(val)
            for j, val in enumerate(pm["singular_values"]): row[f"density_sv_{j+1}"] = float(val)
            rows.append(row)
    return pd.DataFrame(rows)


def unique_endpoints_for_element(results, element):
    out, seen = [], set()
    for res in results:
        for ep in res["endpoints"]:
            if ep["element"] != element: continue
            key = (res["pair"], ep["partner"])
            if ep["partner"] == element and key in seen: continue
            seen.add(key)
            out.append({"result": res, "endpoint": ep})
    return out


def projector_coherence(vectors):
    d = len(vectors[0])
    c = np.zeros((d, d))
    for v in vectors:
        v = np.asarray(v, dtype=float); v = v / np.linalg.norm(v)
        c += np.outer(v, v)
    c /= len(vectors)
    w, u = np.linalg.eigh((c + c.T) * 0.5)
    order = np.argsort(w)[::-1]
    return float(w[order][0]), w[order], u[:, order]


def partner_summary(results) -> pd.DataFrame:
    rows = []
    for element in ELEMENTS:
        items = unique_endpoints_for_element(results, element)
        fblocks = [np.asarray(x["endpoint"]["fock_cross_valence"]) for x in items]
        pblocks = [np.asarray(x["endpoint"]["density_cross_valence"]) for x in items]
        fm = sv_metrics(np.concatenate(fblocks, axis=1))
        pm = sv_metrics(np.concatenate(pblocks, axis=1))
        fcoh, _, _ = projector_coherence([x["endpoint"]["dominant_fock_port"] for x in items])
        pcoh, _, _ = projector_coherence([x["endpoint"]["dominant_density_port"] for x in items])
        acoh, acoh_spec, _ = projector_coherence([x["endpoint"]["active_local_vector"] for x in items])
        occs = np.array([x["endpoint"]["local_density_eigenvalues"] for x in items], dtype=float)
        eps, tabs, sigma0 = [], [], []
        for x in items:
            res, ep = x["result"], x["endpoint"]
            eps.append(res["two_state"]["epsilon_a"] if ep["atom_id"] == 0 else res["two_state"]["epsilon_b"])
            tabs.append(abs(res["two_state"]["t_ab"]))
            sigma0.append(ep["self_energies"]["vacuum"]["sigma_fro"])
        row = {
            "element": element, "n_partners": len(items), "valence_dim": len(fblocks[0]),
            "joint_fock_rank_rel_1e-2": fm["rank_rel_0.01"],
            "joint_fock_rank_rel_1e-3": fm["rank_rel_0.001"],
            "joint_fock_first_sq_fraction": fm["first_sq_fraction"],
            "joint_fock_stable_rank": fm["stable_rank"],
            "joint_density_rank_rel_1e-2": pm["rank_rel_0.01"],
            "joint_density_rank_rel_1e-3": pm["rank_rel_0.001"],
            "joint_density_first_sq_fraction": pm["first_sq_fraction"],
            "joint_density_stable_rank": pm["stable_rank"],
            "dominant_fock_port_coherence": fcoh,
            "dominant_density_port_coherence": pcoh,
            "active_local_vector_coherence": acoh,
            "two_state_epsilon_mean_ev": float(np.mean(eps) * HARTREE_TO_EV),
            "two_state_epsilon_std_ev": float(np.std(eps) * HARTREE_TO_EV),
            "two_state_epsilon_range_ev": float(np.ptp(eps) * HARTREE_TO_EV),
            "two_state_t_mean_ev": float(np.mean(tabs) * HARTREE_TO_EV),
            "two_state_t_range_ev": float(np.ptp(tabs) * HARTREE_TO_EV),
            "sigma_vacuum_mean": float(np.mean(sigma0)),
            "sigma_vacuum_range": float(np.ptp(sigma0)),
        }
        for j in range(occs.shape[1]):
            row[f"local_occ_{j+1}_mean"] = float(np.mean(occs[:, j]))
            row[f"local_occ_{j+1}_std"] = float(np.std(occs[:, j]))
            row[f"local_occ_{j+1}_range"] = float(np.ptp(occs[:, j]))
        for j, val in enumerate(fm["singular_values"]): row[f"joint_fock_sv_{j+1}"] = float(val)
        for j, val in enumerate(pm["singular_values"]): row[f"joint_density_sv_{j+1}"] = float(val)
        for j, val in enumerate(acoh_spec): row[f"active_coherence_eig_{j+1}"] = float(val)
        rows.append(row)
    return pd.DataFrame(rows)


def molecule_summary(results) -> pd.DataFrame:
    return pd.DataFrame([{
        "pair": r["pair"], "r_ang": r["r_ang"], "e_tot": r["e_tot"],
        "nao": r["nao"], "nelectron": r["nelectron"], "nocc": r["nocc"], "nvirt": r["nvirt"],
        "homo_ev": r["homo"] * HARTREE_TO_EV, "lumo_ev": r["lumo"] * HARTREE_TO_EV,
        "gap_ev": r["gap"] * HARTREE_TO_EV,
        "density_idempotency_relerr": r["density_idempotency_relerr"],
        "two_state_capture": r["two_state"]["projection_capture"],
        "two_state_condition": r["two_state"]["projection_condition"],
        "epsilon_a_ev": r["two_state"]["epsilon_a"] * HARTREE_TO_EV,
        "epsilon_b_ev": r["two_state"]["epsilon_b"] * HARTREE_TO_EV,
        "t_ab_ev": r["two_state"]["t_ab"] * HARTREE_TO_EV,
    } for r in results])


def write_report(outdir, mol_df, ep_df, partner_df):
    lines = [
        "# PySCF/MINAO validation of low-rank A–B coupling and dressed-A structure", "",
        "## Protocol", "",
        "RHF/MINAO was applied to all 15 unordered H/F/Cl/Br/I diatomics. Bond lengths were obtained by a one-dimensional RHF/MINAO scan. Matrix analyses use a meta-Löwdin orthogonal AO representation. The spin-summed RHF density matrix is denoted by $P$ and satisfies $P^2=2P$.", "",
        "## Immediate algebraic control", "",
        f"The largest global density idempotency residual is `{float(mol_df['density_idempotency_relerr'].max()):.3e}`. The largest endpoint block-identity residual for $P_{{AB}}P_{{BA}}=2P_A-P_A^2$ is `{float(ep_df['projector_identity_relerr'].max()):.3e}`.", "",
    ]
    nvirt_values = sorted(set(int(x) for x in mol_df["nvirt"]))
    if nvirt_values == [1]:
        lines.append("Every MINAO calculation has exactly one virtual spatial orbital. This forces the occupied-projector cross block to have rank at most one. Rank-one $P_{AB}$ here is therefore an exact minimal-basis consequence, not by itself proof of a universal chemical law.")
    else:
        lines.append(f"The observed numbers of virtual spatial orbitals are {nvirt_values}; low projector rank must be interpreted together with this finite-basis constraint.")
    lines += ["", "## Pair-level rank comparison", ""]
    cols = ["pair", "element", "partner", "valence_dim", "fock_rank_rel_1e-2", "fock_first_sq_fraction", "density_rank_rel_1e-2", "density_first_sq_fraction", "active_local_occupation", "active_density_port_overlap"]
    lines.append(ep_df[cols].round(6).to_markdown(index=False))
    lines += ["", "## Cross-partner summary", ""]
    cols2 = ["element", "valence_dim", "joint_fock_rank_rel_1e-2", "joint_fock_first_sq_fraction", "joint_density_rank_rel_1e-2", "joint_density_first_sq_fraction", "active_local_vector_coherence", "two_state_epsilon_std_ev", "two_state_t_range_ev"]
    lines.append(partner_df[cols2].round(6).to_markdown(index=False))
    lines += ["", "## Interpretation guide", "",
        "1. A low rank of the raw valence Fock block supports the literal $H_{AB}$ hypothesis. A full Fock rank together with rank-one $P_{AB}$ means that the occupied bond/entanglement channel is low rank while the Hamiltonian contains additional spectator and Pauli channels.",
        "2. For a halogen valence space, three local natural occupations near 2 and one fractional occupation identify a stable lone-pair spectator sector plus one partner-sensitive bonding port.",
        "3. `active_local_vector_coherence` is the leading eigenvalue of the average projector onto the partner-dependent active A-side vector. Values near 1 indicate a common A-side port across partners.",
        "4. The two-state effective onsite energies are basis- and state-selection-dependent diagnostics, not observable atomic energies.", "",
        "## Files", "", "- `molecule_summary.csv`", "- `endpoint_metrics.csv`", "- `partner_summary.csv`", "- `raw_results.json`", "- `scan_curves.json`"]
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--basis", default="minao")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results, scan_payload, failures = [], {}, []
    pairs = list(itertools.combinations_with_replacement(ELEMENTS, 2))
    for ipair, (a, b) in enumerate(pairs, 1):
        pair = f"{a}-{b}"
        print(f"[{ipair:02d}/{len(pairs):02d}] {pair}: scanning RHF/{args.basis}", flush=True)
        try:
            r_best, e_best, curve = scan_equilibrium(a, b, args.basis)
            print(f"  R = {r_best:.6f} Ang, E = {e_best:.12f} Eh", flush=True)
            scan_payload[pair] = {"r_best": r_best, "e_best": e_best, "curve": curve}
            results.append(analyze_pair(a, b, r_best, args.basis))
        except Exception as exc:
            traceback.print_exc()
            failures.append({"pair": pair, "error": repr(exc)})
    if failures:
        (args.output / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    if len(results) != len(pairs):
        raise RuntimeError(f"Only {len(results)}/{len(pairs)} calculations succeeded; failures={failures}")
    mol_df = molecule_summary(results)
    ep_df = endpoint_rows(results)
    partner_df = partner_summary(results)
    mol_df.to_csv(args.output / "molecule_summary.csv", index=False)
    ep_df.to_csv(args.output / "endpoint_metrics.csv", index=False)
    partner_df.to_csv(args.output / "partner_summary.csv", index=False)
    (args.output / "scan_curves.json").write_text(json.dumps(jsonable(scan_payload), indent=2), encoding="utf-8")
    (args.output / "raw_results.json").write_text(json.dumps(jsonable(results), indent=2), encoding="utf-8")
    write_report(args.output, mol_df, ep_df, partner_df)
    print("\n=== MOLECULE SUMMARY ===")
    print(mol_df.to_csv(index=False))
    print("\n=== ENDPOINT METRICS ===")
    print(ep_df.to_csv(index=False))
    print("\n=== PARTNER SUMMARY ===")
    print(partner_df.to_csv(index=False))
    print("\n=== REPORT ===")
    print((args.output / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

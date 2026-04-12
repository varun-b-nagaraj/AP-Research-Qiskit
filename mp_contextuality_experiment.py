#!/usr/bin/env python3
"""
Mermin–Peres contextuality experiment driver
--------------------------------------------

What this script does:
- Builds two matched circuit families:
    1) contextuality-preserving
    2) baseline (same logical prep, extra structure-breaking/noise-sensitive overhead)
- Tests them under 3 noise levels: low / medium / high
- Uses 8192 shots per circuit configuration
- Computes:
    * fidelity-like overlap with ideal parity distributions
    * trace distance to ideal parity distributions
    * normalized Peres-Mermin contextuality violation score
- Applies manual zero-noise extrapolation (ZNE) with global gate folding
- Saves raw results + summary CSV + plots

Notes:
- This is a faithful implementation of the *methodology* in your paper,
  but not a guaranteed exact reproduction of unpublished source code.
- The experiment is implemented as the 3x3 Mermin-Peres magic-square game on two EPR pairs.
- Contextuality score is normalized from magic-square win rate using the classical bound 8/9
  and the ideal quantum value 1.
"""

from __future__ import annotations

import argparse
import os
import math
import json
import time
import random
from functools import lru_cache
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.quantum_info import Statevector, Pauli
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
)


def load_simple_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


load_simple_dotenv()

# =========================
# Configuration
# =========================

SEED = 7
OUTDIR = Path("mp_contextuality_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Backends:
USE_REAL_BACKEND = env_bool("MP_USE_REAL_BACKEND", False)
USE_BACKEND_MIMIC = env_bool("MP_USE_BACKEND_MIMIC", False)
IBM_BACKEND_NAME = os.getenv("MP_BACKEND_NAME", "ibm_brisbane")

# Conservative transpilation to preserve circuit structure
OPTIMIZATION_LEVEL = 0

# Defaults follow the paper and can be overridden at runtime.
DEFAULT_SHOTS = 8192

# ZNE noise scale factors (odd integers are standard for folding)
DEFAULT_ZNE_SCALE_FACTORS = [1, 3, 5]

# Noise levels for fully local runs if not mimicking a backend
NOISE_LEVELS = {
    "low": {
        "single_qubit_depol": 0.001,
        "two_qubit_depol": 0.010,
        "readout_p01": 0.010,
        "readout_p10": 0.010,
    },
    "medium": {
        "single_qubit_depol": 0.003,
        "two_qubit_depol": 0.020,
        "readout_p01": 0.020,
        "readout_p10": 0.020,
    },
    "high": {
        "single_qubit_depol": 0.006,
        "two_qubit_depol": 0.040,
        "readout_p01": 0.040,
        "readout_p10": 0.040,
    },
}

# The Mermin-Peres magic-square game observable table.
# Rows have product +I and columns have product -I, matching the game parity rules:
# Alice returns a row with even parity, Bob returns a column with odd parity.
MAGIC_SQUARE = [
    ["XI", "XX", "IX"],
    ["-XZ", "YY", "-ZX"],
    ["IZ", "ZZ", "ZI"],
]

ROW_CONTEXTS = {
    "R1": MAGIC_SQUARE[0],
    "R2": MAGIC_SQUARE[1],
    "R3": MAGIC_SQUARE[2],
}

COLUMN_CONTEXTS = {
    "C1": [MAGIC_SQUARE[row][0] for row in range(3)],
    "C2": [MAGIC_SQUARE[row][1] for row in range(3)],
    "C3": [MAGIC_SQUARE[row][2] for row in range(3)],
}

QUERY_CONFIGS = {
    f"R{row + 1}C{col + 1}": (row, col)
    for row in range(3)
    for col in range(3)
}

NONCONTEXTUAL_WIN_BOUND = 8.0 / 9.0
IDEAL_QUANTUM_WIN_RATE = 1.0

DEFAULT_NOISE_LEVEL_ORDER = ["low", "medium", "high"]
REAL_BACKEND_NOISE_LABEL = "hardware"
DEFAULT_CIRCUIT_TYPES = ["contextuality-preserving", "baseline"]
DEFAULT_MITIGATION_MODES = [False, True]


# =========================
# Utilities
# =========================

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


@lru_cache(maxsize=1)
def ensure_runtime_service():
    """
    Import IBM Runtime lazily so local Aer runs do not pay the import cost or
    get blocked on optional dependency initialization.
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except Exception as exc:
        raise RuntimeError(
            "qiskit-ibm-runtime is not available. Install/configure it or disable "
            "real-backend options."
        ) from exc
    token = os.getenv("IBM_QUANTUM_TOKEN")
    instance = os.getenv("IBM_QUANTUM_INSTANCE")
    channel = os.getenv("IBM_QUANTUM_CHANNEL", "ibm_quantum_platform")

    kwargs = {"channel": channel}
    if token:
        kwargs["token"] = token
    if instance:
        kwargs["instance"] = instance
    return QiskitRuntimeService(**kwargs)


@lru_cache(maxsize=4)
def get_reference_backend(backend_name: Optional[str] = None):
    """Optional backend used either directly or to mimic a noise profile."""
    service = ensure_runtime_service()
    target_backend = backend_name or IBM_BACKEND_NAME
    if target_backend == "auto":
        candidates = []
        for backend in service.backends(simulator=False):
            try:
                status = backend.status()
            except Exception:
                continue
            if not getattr(status, "operational", False):
                continue
            candidates.append((status.pending_jobs, backend.name, backend))
        if not candidates:
            raise RuntimeError("No operational IBM hardware backends are available for this account.")
        candidates.sort(key=lambda item: (item[0], item[1]))
        pending_jobs, target_backend, backend = candidates[0]
        print(f"[backend] auto-selected {target_backend} (pending_jobs={pending_jobs})", flush=True)
        return backend
    backend = service.backend(target_backend)
    try:
        status = backend.status()
        print(f"[backend] selected {backend.name} (pending_jobs={status.pending_jobs}, status={status.status_msg})", flush=True)
    except Exception:
        print(f"[backend] selected {backend.name}", flush=True)
    return backend


def build_local_noise_model(level: str) -> NoiseModel:
    params = NOISE_LEVELS[level]
    noise_model = NoiseModel()

    err_1q = depolarizing_error(params["single_qubit_depol"], 1)
    err_2q = depolarizing_error(params["two_qubit_depol"], 2)
    ro_err = ReadoutError([
        [1 - params["readout_p01"], params["readout_p01"]],
        [params["readout_p10"], 1 - params["readout_p10"]],
    ])

    # Apply to common gate names used after transpilation.
    one_qubit_gates = ["id", "rz", "sx", "x", "u", "u1", "u2", "u3", "h", "s", "sdg"]
    two_qubit_gates = ["cx", "ecr", "cz"]

    for g in one_qubit_gates:
        try:
            noise_model.add_all_qubit_quantum_error(err_1q, [g])
        except Exception:
            pass
    for g in two_qubit_gates:
        try:
            noise_model.add_all_qubit_quantum_error(err_2q, [g])
        except Exception:
            pass

    noise_model.add_all_qubit_readout_error(ro_err)
    return noise_model


def build_simulator(level: str):
    """
    Build either:
      - a local synthetic-noise simulator, or
      - a backend-mimicking simulator based on a real IBM backend.
    """
    if USE_BACKEND_MIMIC:
        backend = get_reference_backend(IBM_BACKEND_NAME)
        sim = AerSimulator.from_backend(backend)
        return sim, backend
    else:
        sim = AerSimulator(
            noise_model=build_local_noise_model(level),
            seed_simulator=SEED,
        )
        return sim, None


def run_compiled_circuits(backend, compiled_circuits: List[QuantumCircuit], shots: int) -> List[Dict[str, int]]:
    """
    Submit a batch of compiled circuits to a backend and return one counts dict per circuit.
    """
    job = backend.run(compiled_circuits, shots=shots)
    result = job.result()
    return [result.get_counts(i) for i in range(len(compiled_circuits))]


def run_sampler_circuits(backend, compiled_circuits: List[QuantumCircuit], shots: int) -> List[Dict[str, int]]:
    """
    Submit a batch of circuits to IBM Runtime SamplerV2 and return one counts dict
    per circuit. This is the supported interface for real hardware execution.
    """
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(
        mode=backend,
        options={
            "default_shots": shots,
            "dynamical_decoupling": {
                "enable": True,
                "sequence_type": "XpXm",
            },
            "twirling": {
                "enable_gates": True,
            },
        },
    )
    job = sampler.run(compiled_circuits, shots=shots)
    print(f"[hardware] sampler job submitted: {job.job_id()}", flush=True)
    result = job.result()

    counts_list: List[Dict[str, int]] = []
    for pub_result in result:
        data_values = list(pub_result.data.values())
        if not data_values:
            raise RuntimeError("Sampler result did not contain classical measurement data.")
        counts_list.append(data_values[0].get_counts())
    return counts_list


def pauli_basis_rotation(qc: QuantumCircuit, qubit: int, pauli: str) -> None:
    """
    Rotate measurement basis so measuring Z is equivalent to measuring the requested Pauli.
    """
    if pauli == "I":
        return
    if pauli == "X":
        qc.h(qubit)
    elif pauli == "Y":
        qc.sdg(qubit)
        qc.h(qubit)
    elif pauli == "Z":
        return
    else:
        raise ValueError(f"Unsupported pauli: {pauli}")


def bit_to_eigenvalue(bit: str) -> int:
    # In Z basis: |0> -> +1, |1> -> -1
    return +1 if bit == "0" else -1


def counts_to_probs(counts: Dict[str, int], shots: int) -> Dict[str, float]:
    return {k: v / shots for k, v in counts.items()}


def l1_trace_distance(
    p: Dict[str, float],
    q: Dict[str, float],
    keys: Iterable[str],
) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def classical_fidelity(
    p: Dict[str, float],
    q: Dict[str, float],
    keys: Iterable[str],
) -> float:
    # Bhattacharyya-style fidelity between classical distributions
    return (sum(math.sqrt(max(p.get(k, 0.0), 0.0) * max(q.get(k, 0.0), 0.0)) for k in keys)) ** 2


def richardson_extrapolate(x_vals: List[float], y_vals: List[float]) -> float:
    """
    Richardson-style extrapolation to zero noise by polynomial fit in noise scale.
    For 3 points, quadratic interpolation is robust enough for this workflow.
    """
    coeffs = np.polyfit(x_vals, y_vals, deg=min(2, len(x_vals) - 1))
    poly = np.poly1d(coeffs)
    y0 = float(poly(0.0))
    return y0


# =========================
# Circuit construction
# =========================

def build_base_state_circuit() -> QuantumCircuit:
    """
    Prepare the two-EPR-pair resource state for the magic-square game.
    Qubits 0-1 belong to Alice and 2-3 belong to Bob.
    """
    qc = QuantumCircuit(4, 4)
    qc.h(0)
    qc.cx(0, 2)
    qc.h(1)
    qc.cx(1, 3)
    return qc


def build_contextuality_preserving_prep() -> QuantumCircuit:
    """
    Low-depth circuit intended to preserve contextual correlations by avoiding
    unnecessary overhead before measurement contexts are applied.
    """
    qc = build_base_state_circuit()
    # Keep it minimal by design.
    return qc


def build_baseline_prep() -> QuantumCircuit:
    """
    Baseline matched circuit:
    - same logical state intent
    - extra Clifford overhead / identity-equivalent structure
    - more vulnerable to noise after transpilation
    """
    qc = build_base_state_circuit()

    # Identity-equivalent overhead that increases exposure to noise without
    # changing the ideal state.
    qc.barrier()
    qc.s(0)
    qc.sdg(0)
    qc.h(1)
    qc.h(1)
    qc.x(2)
    qc.x(2)
    qc.s(3)
    qc.sdg(3)
    qc.cx(0, 2)
    qc.cx(0, 2)
    qc.cx(1, 3)
    qc.cx(1, 3)
    qc.barrier()

    return qc


def signed_pauli_matrix(label: str) -> np.ndarray:
    sign = -1 if label.startswith("-") else 1
    pauli_label = label[1:] if label.startswith("-") else label
    return sign * Pauli(pauli_label).to_matrix()


@lru_cache(maxsize=None)
def get_local_measurement_spec(observables: Tuple[str, str, str]) -> Dict[str, object]:
    """
    Construct a simultaneous eigenbasis for a commuting 3-observable local context.
    """
    operators = [signed_pauli_matrix(obs) for obs in observables]
    combo = sum((i + 1) * op for i, op in enumerate(operators))
    _, eigenvectors = np.linalg.eigh(combo)

    eigen_data: List[Tuple[Tuple[int, int, int], np.ndarray]] = []
    for idx in range(eigenvectors.shape[1]):
        vec = eigenvectors[:, idx]
        observable_eigenvalues = []
        for op in operators:
            ev = np.vdot(vec, op @ vec)
            ev = np.real_if_close(ev)
            observable_eigenvalues.append(int(round(float(np.real(ev)))))
        eigen_data.append((tuple(observable_eigenvalues), vec))

    # Stable ordering for deterministic bitstring decoding.
    eigen_data.sort(key=lambda item: item[0])
    measurement_unitary = np.column_stack([vec for _, vec in eigen_data])
    outcome_eigenvalues = {
        format(index, "02b"): eigenvalues
        for index, (eigenvalues, _) in enumerate(eigen_data)
    }

    return {
        "measurement_unitary": measurement_unitary,
        "measurement_unitary_dagger": measurement_unitary.conj().T,
        "outcome_eigenvalues": outcome_eigenvalues,
        "outcome_labels": list(outcome_eigenvalues.keys()),
    }


def parse_query_name(query_name: str) -> Tuple[int, int]:
    try:
        return QUERY_CONFIGS[query_name]
    except KeyError as exc:
        raise ValueError(f"Unknown query configuration: {query_name}") from exc


def get_row_measurement_spec(row_index: int) -> Dict[str, object]:
    return get_local_measurement_spec(tuple(MAGIC_SQUARE[row_index]))


def get_column_measurement_spec(col_index: int) -> Dict[str, object]:
    return get_local_measurement_spec(tuple(MAGIC_SQUARE[row][col_index] for row in range(3)))


def build_query_circuit(
    prep: QuantumCircuit,
    query_name: str,
    include_measurements: bool,
) -> QuantumCircuit:
    row_index, col_index = parse_query_name(query_name)
    qc = prep.copy()
    qc.name = f"{prep.name}_{query_name}" if prep.name else query_name

    row_spec = get_row_measurement_spec(row_index)
    col_spec = get_column_measurement_spec(col_index)

    qc.unitary(row_spec["measurement_unitary_dagger"], [0, 1], label=f"row_{row_index + 1}_meas")
    qc.unitary(col_spec["measurement_unitary_dagger"], [2, 3], label=f"col_{col_index + 1}_meas")

    if include_measurements:
        for qubit in range(4):
            qc.measure(qubit, qubit)
    return qc


def fold_global(circuit: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    """
    Simple global folding for ZNE:
      U -> U (U^dagger U)^k
    where scale_factor = 2k+1.
    """
    if scale_factor < 1 or scale_factor % 2 == 0:
        raise ValueError("scale_factor must be an odd positive integer")

    if scale_factor == 1:
        return circuit.copy()

    k = (scale_factor - 1) // 2
    folded = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    folded.compose(circuit, inplace=True)

    unitary_part = circuit.remove_final_measurements(inplace=False)
    for _ in range(k):
        folded.compose(unitary_part.inverse(), inplace=True)
        folded.compose(unitary_part, inplace=True)

    # Re-add measurements if they existed in the original circuit.
    if circuit.num_clbits > 0:
        meas_circ = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        for instruction in circuit.data:
            if instruction.operation.name == "measure":
                meas_circ.append(
                    instruction.operation,
                    instruction.qubits,
                    instruction.clbits,
                )
        folded.compose(meas_circ, inplace=True)

    return folded


# =========================
# Context evaluation
# =========================

def decode_query_outcome(
    query_name: str,
    bitstring: str,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Decode a 4-bit outcome into Alice's row outputs and Bob's column outputs.
    """
    bits = bitstring.replace(" ", "")
    bits = bits[-4:].zfill(4)
    row_index, col_index = parse_query_name(query_name)
    row_bits = bits[-2:]   # c1c0
    col_bits = bits[:2]    # c3c2

    row_values = get_row_measurement_spec(row_index)["outcome_eigenvalues"][row_bits]
    col_values = get_column_measurement_spec(col_index)["outcome_eigenvalues"][col_bits]
    return row_values, col_values


def query_win_from_bitstring(query_name: str, bitstring: str) -> int:
    row_index, col_index = parse_query_name(query_name)
    row_values, col_values = decode_query_outcome(query_name, bitstring)
    row_ok = np.prod(row_values) == +1
    col_ok = np.prod(col_values) == -1
    intersection_ok = row_values[col_index] == col_values[row_index]
    return int(row_ok and col_ok and intersection_ok)


def outcome_distribution_from_counts(
    counts: Dict[str, int],
    shots: int,
) -> Dict[str, float]:
    """
    Return the probability distribution over the sixteen 4-bit game outcomes.
    """
    dist: Dict[str, float] = {}
    for bitstring, c in counts.items():
        bits = bitstring.replace(" ", "")[-4:].zfill(4)
        dist[bits] = dist.get(bits, 0.0) + (c / shots)
    for label in [format(index, "04b") for index in range(16)]:
        dist.setdefault(label, 0.0)
    return dist


def win_distribution_from_outcome_distribution(
    query_name: str,
    outcome_dist: Dict[str, float],
) -> Dict[str, float]:
    win = 0.0
    lose = 0.0
    for bitstring, prob in outcome_dist.items():
        if query_win_from_bitstring(query_name, bitstring) == 1:
            win += prob
        else:
            lose += prob
    return {"win": win, "lose": lose}


def ideal_outcome_distribution(
    prep_circuit: QuantumCircuit,
    query_name: str,
) -> Dict[str, float]:
    """
    Ideal theoretical distribution for a magic-square query pair obtained by
    noiseless statevector simulation.
    """
    ideal_query = build_query_circuit(prep_circuit, query_name, include_measurements=False)
    state = Statevector.from_instruction(ideal_query)
    probs = state.probabilities_dict()
    ideal_dist = {format(index, "04b"): 0.0 for index in range(16)}
    for bitstring, prob in probs.items():
        bits = bitstring.replace(" ", "")[-4:].zfill(4)
        ideal_dist[bits] += float(prob)
    return ideal_dist


def witness_from_expectations(context_expectations: Dict[str, float]) -> float:
    """
    Raw contextuality witness taken as the average magic-square game win rate
    across all 9 query pairs.
    """
    return float(np.mean(list(context_expectations.values())))


def normalized_contextuality_score(witness: float) -> float:
    """
    Normalize witness violation into [0, 1]:
      score = max(0, min(1, (win_rate - 8/9) / (1 - 8/9)))
    """
    return float(
        np.clip(
            (witness - NONCONTEXTUAL_WIN_BOUND) / (IDEAL_QUANTUM_WIN_RATE - NONCONTEXTUAL_WIN_BOUND),
            0.0,
            1.0,
        )
    )


# =========================
# Execution
# =========================

@dataclass
class RunRecord:
    circuit_type: str
    noise_level: str
    mitigated: bool
    backend: str
    shots_per_context: int
    zne_scale_factors: str
    fidelity: float
    trace_distance: float
    contextuality_violation: float
    win_probability: float


@dataclass
class ContextRecord:
    circuit_type: str
    noise_level: str
    mitigated: bool
    query_name: str
    shots: int
    row_index: int
    column_index: int
    win_probability: float
    fidelity: float
    trace_distance: float
    success_probability: float
    failure_probability: float
    raw_scale_factor_dists: str


@dataclass
class ExperimentConfig:
    shots: int
    zne_scale_factors: List[int]
    noise_levels: List[str]
    circuit_types: List[str]
    mitigation_modes: List[bool]
    contexts: List[str]
    outdir: Path
    use_real_backend: bool
    use_backend_mimic: bool
    backend_name: str
    smoke_test: bool


def compile_circuit(
    qc: QuantumCircuit,
    backend_or_sim,
) -> QuantumCircuit:
    """
    Conservative compilation to preserve logical structure as much as possible.
    """
    try:
        pm = generate_preset_pass_manager(
            optimization_level=OPTIMIZATION_LEVEL,
            backend=backend_or_sim,
        )
        compiled = pm.run(qc)
        return compiled
    except Exception:
        # Fallback
        return transpile(qc, backend_or_sim, optimization_level=OPTIMIZATION_LEVEL, seed_transpiler=SEED)


def run_single_context(
    prep_circuit: QuantumCircuit,
    circuit_type: str,
    context_name: str,
    simulator,
    shots: int,
    zne_scale_factors: Optional[List[int]] = None,
    noise_level: Optional[str] = None,
    mitigated: Optional[bool] = None,
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    """
    Run one context, optionally with ZNE.
    Returns:
      - extrapolated or direct joint outcome distribution
      - raw joint outcome distributions by scale factor
    """
    base_meas = build_query_circuit(prep_circuit, context_name, include_measurements=True)
    raw_outcome_dists: Dict[int, Dict[str, float]] = {}
    progress_prefix = (
        f"[run] circuit={circuit_type} noise={noise_level or 'n/a'} "
        f"mitigated={mitigated if mitigated is not None else 'n/a'} query={context_name}"
    )

    if not zne_scale_factors:
        print(progress_prefix, flush=True)
        compiled = compile_circuit(base_meas, simulator)
        result = simulator.run(compiled, shots=shots).result()
        counts = result.get_counts()
        dist = outcome_distribution_from_counts(counts, shots)
        return dist, raw_outcome_dists

    # Run folded circuits
    outcome_labels = tuple(format(index, "04b") for index in range(16))
    outcome_probs_by_label = {label: [] for label in outcome_labels}

    for sf in zne_scale_factors:
        print(f"{progress_prefix} scale_factor={sf}", flush=True)
        folded = fold_global(base_meas, sf)
        compiled = compile_circuit(folded, simulator)
        result = simulator.run(compiled, shots=shots).result()
        counts = result.get_counts()
        dist = outcome_distribution_from_counts(counts, shots)
        raw_outcome_dists[sf] = dist
        for label in outcome_labels:
            outcome_probs_by_label[label].append(dist.get(label, 0.0))

    extrapolated = {
        label: max(0.0, richardson_extrapolate(zne_scale_factors, probs))
        for label, probs in outcome_probs_by_label.items()
    }
    s = sum(extrapolated.values())
    if s <= 0:
        dist0 = {label: 0.25 for label in outcome_labels}
    else:
        dist0 = {label: value / s for label, value in extrapolated.items()}

    return dist0, raw_outcome_dists


def aggregate_metrics(
    prep_circuit: QuantumCircuit,
    context_dists: Dict[str, Dict[str, float]],
) -> Tuple[float, float, float, float]:
    """
    Aggregate all 9 row/column query pairs into:
      fidelity, trace_distance, normalized_contextuality_violation, mean win rate
    by comparing each measured outcome distribution to the ideal noiseless query
    distribution and by computing the game success probability.
    """
    context_expectations = {}
    fidelities = []
    trace_distances = []

    for ctx, dist in context_dists.items():
        ideal = ideal_outcome_distribution(prep_circuit, ctx)
        keys = [format(index, "04b") for index in range(16)]
        fidelities.append(classical_fidelity(dist, ideal, keys))
        trace_distances.append(l1_trace_distance(dist, ideal, keys))
        win_dist = win_distribution_from_outcome_distribution(ctx, dist)
        context_expectations[ctx] = win_dist["win"]

    witness = witness_from_expectations(context_expectations)
    contextuality = normalized_contextuality_score(witness)

    return (
        float(np.mean(fidelities)),
        float(np.mean(trace_distances)),
        contextuality,
        witness,
    )


def build_context_records(
    circuit_type: str,
    noise_level: str,
    mitigated: bool,
    shots: int,
    context_dists: Dict[str, Dict[str, float]],
    raw_context_scale_dists: Dict[str, Dict[int, Dict[str, float]]],
    prep_circuit: QuantumCircuit,
) -> List[ContextRecord]:
    records: List[ContextRecord] = []
    for ctx, dist in context_dists.items():
        ideal = ideal_outcome_distribution(prep_circuit, ctx)
        win_dist = win_distribution_from_outcome_distribution(ctx, dist)
        row_index, col_index = parse_query_name(ctx)
        records.append(
            ContextRecord(
                circuit_type=circuit_type,
                noise_level=noise_level,
                mitigated=mitigated,
                query_name=ctx,
                shots=shots,
                row_index=row_index + 1,
                column_index=col_index + 1,
                win_probability=win_dist["win"],
                fidelity=classical_fidelity(dist, ideal, [format(index, "04b") for index in range(16)]),
                trace_distance=l1_trace_distance(dist, ideal, [format(index, "04b") for index in range(16)]),
                success_probability=win_dist["win"],
                failure_probability=win_dist["lose"],
                raw_scale_factor_dists=json.dumps(raw_context_scale_dists.get(ctx, {}), sort_keys=True),
            )
        )
    return records


def run_family(
    config: ExperimentConfig,
    circuit_type: str,
    prep_circuit: QuantumCircuit,
    noise_level: str,
    mitigated: bool,
) -> Tuple[RunRecord, List[ContextRecord]]:
    """
    Execute all selected magic-square query pairs for one circuit family / noise level /
    mitigation setting.
    """
    simulator = None
    ref_backend = None
    backend_name = config.backend_name if config.use_backend_mimic else "local_aer_noise_model"
    if not config.use_real_backend:
        simulator, ref_backend = build_simulator(noise_level)
    raw_context_scale_dists: Dict[str, Dict[int, Dict[str, float]]] = {}

    # Optional real backend path.
    if config.use_real_backend:
        backend = get_reference_backend(config.backend_name)
        backend_name = backend.name
        context_dists = {}

        if mitigated:
            folded_jobs: List[Tuple[str, int, QuantumCircuit]] = []
            for ctx in config.contexts:
                base_meas = build_query_circuit(prep_circuit, ctx, include_measurements=True)
                raw_context_scale_dists[ctx] = {}
                for sf in config.zne_scale_factors:
                    print(
                        f"[run] circuit={circuit_type} noise={noise_level} mitigated={mitigated} "
                        f"query={ctx} scale_factor={sf}",
                        flush=True,
                    )
                    folded = fold_global(base_meas, sf)
                    compiled = compile_circuit(folded, backend)
                    folded_jobs.append((ctx, sf, compiled))

            counts_list = run_sampler_circuits(
                backend,
                [compiled for _, _, compiled in folded_jobs],
                shots=config.shots,
            )
            for (ctx, sf, _), counts in zip(folded_jobs, counts_list):
                raw_context_scale_dists[ctx][sf] = outcome_distribution_from_counts(counts, config.shots)

            outcome_labels = tuple(format(index, "04b") for index in range(16))
            for ctx in config.contexts:
                extrapolated = {}
                for label in outcome_labels:
                    extrapolated[label] = max(
                        0.0,
                        richardson_extrapolate(
                            config.zne_scale_factors,
                            [raw_context_scale_dists[ctx][sf].get(label, 0.0) for sf in config.zne_scale_factors],
                        ),
                    )
                s = sum(extrapolated.values())
                if s <= 0:
                    context_dists[ctx] = {label: 0.25 for label in outcome_labels}
                else:
                    context_dists[ctx] = {label: value / s for label, value in extrapolated.items()}
        else:
            compiled_jobs: List[Tuple[str, QuantumCircuit]] = []
            for ctx in config.contexts:
                print(f"[run] circuit={circuit_type} noise={noise_level} mitigated={mitigated} query={ctx}", flush=True)
                base_meas = build_query_circuit(prep_circuit, ctx, include_measurements=True)
                compiled_jobs.append((ctx, compile_circuit(base_meas, backend)))

            counts_list = run_sampler_circuits(
                backend,
                [compiled for _, compiled in compiled_jobs],
                shots=config.shots,
            )
            for (ctx, _), counts in zip(compiled_jobs, counts_list):
                context_dists[ctx] = outcome_distribution_from_counts(counts, config.shots)
    else:
        context_dists = {}
        for ctx in config.contexts:
            dist, raw_dists = run_single_context(
                prep_circuit=prep_circuit,
                circuit_type=circuit_type,
                context_name=ctx,
                simulator=simulator,
                shots=config.shots,
                zne_scale_factors=config.zne_scale_factors if mitigated else None,
                noise_level=noise_level,
                mitigated=mitigated,
            )
            context_dists[ctx] = dist
            raw_context_scale_dists[ctx] = raw_dists

    fidelity, trace_distance, contextuality, win_probability = aggregate_metrics(prep_circuit, context_dists)

    run_record = RunRecord(
        circuit_type=circuit_type,
        noise_level=noise_level,
        mitigated=mitigated,
        backend=backend_name,
        shots_per_context=config.shots,
        zne_scale_factors=json.dumps(config.zne_scale_factors if mitigated else [1]),
        fidelity=fidelity,
        trace_distance=trace_distance,
        contextuality_violation=contextuality,
        win_probability=win_probability,
    )
    context_records = build_context_records(
        circuit_type=circuit_type,
        noise_level=noise_level,
        mitigated=mitigated,
        shots=config.shots,
        context_dists=context_dists,
        raw_context_scale_dists=raw_context_scale_dists,
        prep_circuit=prep_circuit,
    )
    return run_record, context_records


# =========================
# Plotting and output
# =========================

def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def save_excel_workbook(
    outpath: Path,
    summary_df: pd.DataFrame,
    query_df: pd.DataFrame,
    summary_tables: Dict[str, pd.DataFrame],
    manifest: Dict,
) -> Tuple[bool, str]:
    """
    Write a single Excel workbook containing the raw outputs and all derived tables.
    """
    manifest_df = pd.DataFrame(
        [{"key": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value} for key, value in manifest.items()]
    )

    engine = None
    for candidate in ("openpyxl", "xlsxwriter"):
        try:
            __import__(candidate)
            engine = candidate
            break
        except Exception:
            continue

    if engine is None:
        return False, "Excel export skipped: install `openpyxl` or `xlsxwriter` in the active environment."

    with pd.ExcelWriter(outpath, engine=engine) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        query_df.to_excel(writer, sheet_name="queries", index=False)
        manifest_df.to_excel(writer, sheet_name="manifest", index=False)
        summary_tables["paper_summary"].to_excel(writer, sheet_name="paper_summary", index=False)
        summary_tables["fidelity_table"].to_excel(writer, sheet_name="fidelity_by_noise")
        summary_tables["win_rate_table"].to_excel(writer, sheet_name="win_rate_by_noise")
        summary_tables["contextuality_table"].to_excel(writer, sheet_name="contextuality_by_noise")
        summary_tables["preserving_advantage"].to_excel(writer, sheet_name="preserving_advantage", index=False)
        summary_tables["mitigation_gain"].to_excel(writer, sheet_name="mitigation_gain", index=False)
        summary_tables["weakest_queries"].to_excel(writer, sheet_name="weakest_queries", index=False)
    return True, f"Saved Excel workbook with engine `{engine}`."


def save_json(payload: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def validate_probability_distributions(context_df: pd.DataFrame) -> None:
    if context_df.empty:
        return
    sums = (context_df["success_probability"] + context_df["failure_probability"]).round(9)
    if not np.allclose(sums.values, np.ones(len(sums)), atol=1e-8):
        bad_rows = context_df.loc[~np.isclose(sums.values, 1.0, atol=1e-8)]
        raise ValueError(f"Non-normalized context distributions detected:\n{bad_rows.to_string(index=False)}")


def ordered_noise_levels(values: Iterable[str]) -> List[str]:
    values = list(dict.fromkeys(values))
    preferred = [level for level in DEFAULT_NOISE_LEVEL_ORDER if level in values]
    extras = [level for level in values if level not in preferred]
    return preferred + extras


def build_interpretable_summary_tables(
    summary_df: pd.DataFrame,
    query_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    summary = summary_df.copy()
    summary["mitigation_label"] = summary["mitigated"].map({False: "pre", True: "post"})
    summary["circuit_short"] = summary["circuit_type"].replace(
        {
            "contextuality-preserving": "preserving",
            "baseline": "baseline",
        }
    )

    paper_summary = (
        summary[
            [
                "noise_level",
                "circuit_type",
                "mitigated",
                "win_probability",
                "contextuality_violation",
                "fidelity",
                "trace_distance",
            ]
        ]
        .sort_values(["noise_level", "mitigated", "circuit_type"])
        .reset_index(drop=True)
    )

    fidelity_table = (
        summary.pivot_table(
            index="noise_level",
            columns=["circuit_short", "mitigation_label"],
            values="fidelity",
        )
        .reindex(ordered_noise_levels(summary["noise_level"]))
        .round(6)
    )

    win_rate_table = (
        summary.pivot_table(
            index="noise_level",
            columns=["circuit_short", "mitigation_label"],
            values="win_probability",
        )
        .reindex(ordered_noise_levels(summary["noise_level"]))
        .round(6)
    )

    contextuality_table = (
        summary.pivot_table(
            index="noise_level",
            columns=["circuit_short", "mitigation_label"],
            values="contextuality_violation",
        )
        .reindex(ordered_noise_levels(summary["noise_level"]))
        .round(6)
    )

    preserving_vs_baseline = (
        summary.pivot_table(
            index=["noise_level", "mitigated"],
            columns="circuit_type",
            values=["win_probability", "fidelity", "trace_distance", "contextuality_violation"],
        )
        .sort_index()
    )
    delta_rows = []
    for (noise_level, mitigated), row in preserving_vs_baseline.iterrows():
        delta_rows.append(
            {
                "noise_level": noise_level,
                "mitigated": mitigated,
                "delta_win_probability": row[("win_probability", "contextuality-preserving")] - row[("win_probability", "baseline")],
                "delta_contextuality": row[("contextuality_violation", "contextuality-preserving")] - row[("contextuality_violation", "baseline")],
                "delta_fidelity": row[("fidelity", "contextuality-preserving")] - row[("fidelity", "baseline")],
                "delta_trace_distance": row[("trace_distance", "contextuality-preserving")] - row[("trace_distance", "baseline")],
            }
        )
    delta_table = pd.DataFrame(delta_rows).round(6)

    mitigation_gain_rows = []
    for circuit_type, sub in summary.groupby("circuit_type"):
        pre = sub[sub["mitigated"] == False].set_index("noise_level")
        post = sub[sub["mitigated"] == True].set_index("noise_level")
        shared = pre.index.intersection(post.index)
        for noise_level in shared:
            mitigation_gain_rows.append(
                {
                    "circuit_type": circuit_type,
                    "noise_level": noise_level,
                    "gain_win_probability": post.loc[noise_level, "win_probability"] - pre.loc[noise_level, "win_probability"],
                    "gain_contextuality": post.loc[noise_level, "contextuality_violation"] - pre.loc[noise_level, "contextuality_violation"],
                    "gain_fidelity": post.loc[noise_level, "fidelity"] - pre.loc[noise_level, "fidelity"],
                    "gain_trace_distance": post.loc[noise_level, "trace_distance"] - pre.loc[noise_level, "trace_distance"],
                }
            )
    mitigation_gain_table = pd.DataFrame(mitigation_gain_rows).round(6)

    query = query_df.copy()
    query["mitigation_label"] = query["mitigated"].map({False: "pre", True: "post"})
    query_weakest = (
        query.sort_values(["noise_level", "mitigated", "win_probability", "trace_distance"], ascending=[True, True, True, False])
        .groupby(["noise_level", "mitigated"])
        .head(3)
        [
            [
                "noise_level",
                "mitigated",
                "query_name",
                "row_index",
                "column_index",
                "win_probability",
                "fidelity",
                "trace_distance",
            ]
        ]
        .reset_index(drop=True)
        .round(6)
    )

    return {
        "paper_summary": paper_summary.round(6),
        "fidelity_table": fidelity_table,
        "win_rate_table": win_rate_table,
        "contextuality_table": contextuality_table,
        "preserving_advantage": delta_table,
        "mitigation_gain": mitigation_gain_table,
        "weakest_queries": query_weakest,
    }


def write_text_report(
    summary_df: pd.DataFrame,
    query_df: pd.DataFrame,
    outpath: Path,
) -> None:
    best = summary_df.sort_values(["win_probability", "fidelity"], ascending=[False, False]).iloc[0]
    worst = summary_df.sort_values(["win_probability", "fidelity"], ascending=[True, True]).iloc[0]
    pre = summary_df[summary_df["mitigated"] == False]
    post = summary_df[summary_df["mitigated"] == True]
    preserving_pre = pre[pre["circuit_type"] == "contextuality-preserving"]["win_probability"].mean()
    baseline_pre = pre[pre["circuit_type"] == "baseline"]["win_probability"].mean()
    preserving_post = post[post["circuit_type"] == "contextuality-preserving"]["win_probability"].mean()
    baseline_post = post[post["circuit_type"] == "baseline"]["win_probability"].mean()

    weakest = query_df.sort_values(["win_probability", "trace_distance"], ascending=[True, False]).head(5)

    lines = [
        "Mermin-Peres Magic-Square Experiment Report",
        "",
        "Headline findings",
        f"- Best condition: {best['circuit_type']} at {best['noise_level']} noise with mitigated={best['mitigated']}, win_probability={best['win_probability']:.6f}, fidelity={best['fidelity']:.6f}, contextuality={best['contextuality_violation']:.6f}",
        f"- Worst condition: {worst['circuit_type']} at {worst['noise_level']} noise with mitigated={worst['mitigated']}, win_probability={worst['win_probability']:.6f}, fidelity={worst['fidelity']:.6f}, contextuality={worst['contextuality_violation']:.6f}",
        f"- Mean pre-mitigation win rate: preserving={preserving_pre:.6f}, baseline={baseline_pre:.6f}",
        f"- Mean post-mitigation win rate: preserving={preserving_post:.6f}, baseline={baseline_post:.6f}",
        "",
        "Weakest query pairs overall",
    ]
    for _, row in weakest.iterrows():
        lines.append(
            f"- {row['query_name']} ({row['circuit_type']}, noise={row['noise_level']}, mitigated={row['mitigated']}): "
            f"win_probability={row['win_probability']:.6f}, fidelity={row['fidelity']:.6f}, trace_distance={row['trace_distance']:.6f}"
        )
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_fidelity_by_noise(df: pd.DataFrame, outpath: Path) -> None:
    noise_order = ordered_noise_levels(df["noise_level"].unique())
    pivot = (
        df[df["mitigated"] == False]
        .pivot(index="noise_level", columns="circuit_type", values="fidelity")
        .reindex(noise_order)
    )

    plt.figure(figsize=(7, 5))
    x = np.arange(len(pivot.index))
    width = 0.4

    cols = list(pivot.columns)
    if len(cols) >= 2:
        plt.bar(x - width / 2, pivot[cols[0]].values, width=width, label=cols[0])
        plt.bar(x + width / 2, pivot[cols[1]].values, width=width, label=cols[1])
    elif len(cols) == 1:
        plt.bar(x, pivot[cols[0]].values, width=width, label=cols[0])

    plt.xticks(x, pivot.index)
    plt.title("Figure 1. Mean Fidelity by Circuit Type and Noise Level")
    plt.xlabel("Noise level")
    plt.ylabel("Mean fidelity")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_win_probability_by_noise(df: pd.DataFrame, outpath: Path) -> None:
    noise_order = ordered_noise_levels(df["noise_level"].unique())
    pivot = (
        df[df["mitigated"] == False]
        .pivot(index="noise_level", columns="circuit_type", values="win_probability")
        .reindex(noise_order)
    )

    plt.figure(figsize=(7, 5))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], marker="o", label=col)
    plt.axhline(NONCONTEXTUAL_WIN_BOUND, color="#444444", linestyle="--", linewidth=1, label="classical bound")
    plt.title("Magic-Square Win Rate vs Noise Level")
    plt.xlabel("Noise level")
    plt.ylabel("Mean win probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_contextuality_by_noise(df: pd.DataFrame, outpath: Path) -> None:
    noise_order = ordered_noise_levels(df["noise_level"].unique())
    pivot = (
        df[df["mitigated"] == False]
        .pivot(index="noise_level", columns="circuit_type", values="contextuality_violation")
        .reindex(noise_order)
    )

    plt.figure(figsize=(7, 5))
    x = np.arange(len(pivot.index))
    width = 0.35

    cols = list(pivot.columns)
    if len(cols) >= 2:
        plt.bar(x - width / 2, pivot[cols[0]].values, width=width, label=cols[0])
        plt.bar(x + width / 2, pivot[cols[1]].values, width=width, label=cols[1])
    else:
        plt.bar(x, pivot[cols[0]].values, width=width, label=cols[0])

    plt.xticks(x, pivot.index)
    plt.title("Mean Contextuality Violation by Circuit Type and Noise Level")
    plt.xlabel("Noise level")
    plt.ylabel("Normalized contextuality violation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_query_heatmaps(query_df: pd.DataFrame, outdir: Path) -> None:
    for metric, title, stem in [
        ("win_probability", "Per-Query Win Probability", "figure_query_win_heatmap"),
        ("trace_distance", "Per-Query Trace Distance", "figure_query_trace_distance_heatmap"),
    ]:
        for mitigated in [False, True]:
            subset = query_df[query_df["mitigated"] == mitigated]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            for ax, circuit_type in zip(axes, ["contextuality-preserving", "baseline"]):
                data = (
                    subset[subset["circuit_type"] == circuit_type]
                    .groupby(["row_index", "column_index"], as_index=False)[metric]
                    .mean()
                    .pivot(index="row_index", columns="column_index", values=metric)
                    .reindex(index=[1, 2, 3], columns=[1, 2, 3])
                )
                im = ax.imshow(data.values, cmap="viridis" if metric == "win_probability" else "magma", aspect="equal")
                ax.set_title(circuit_type)
                ax.set_xlabel("Column query")
                ax.set_ylabel("Row query")
                ax.set_xticks([0, 1, 2], labels=["C1", "C2", "C3"])
                ax.set_yticks([0, 1, 2], labels=["R1", "R2", "R3"])
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, f"{data.values[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
            fig.suptitle(f"{title} ({'post-mitigation' if mitigated else 'pre-mitigation'})")
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
            fig.savefig(outdir / f"{stem}_{'post' if mitigated else 'pre'}.png", dpi=200)
            plt.close(fig)


def plot_contextuality_vs_fidelity(df: pd.DataFrame, outpath: Path) -> None:
    plt.figure(figsize=(7, 5))
    for ct, sub in df.groupby("circuit_type"):
        plt.scatter(sub["contextuality_violation"], sub["fidelity"], label=ct)
    plt.title("Contextuality Violation vs Fidelity")
    plt.xlabel("Normalized contextuality violation")
    plt.ylabel("Fidelity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_pre_post_mitigation(df: pd.DataFrame, outpath: Path) -> None:
    grouped = (
        df.groupby(["circuit_type", "mitigated"], as_index=False)["fidelity"]
        .mean()
        .sort_values(["circuit_type", "mitigated"])
    )

    circuit_types = list(grouped["circuit_type"].unique())
    pre = []
    post = []
    for ct in circuit_types:
        pre_val = grouped[(grouped["circuit_type"] == ct) & (grouped["mitigated"] == False)]["fidelity"].mean()
        post_val = grouped[(grouped["circuit_type"] == ct) & (grouped["mitigated"] == True)]["fidelity"].mean()
        pre.append(pre_val)
        post.append(post_val)

    x = np.arange(len(circuit_types))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, pre, width=width, label="Pre-mitigation")
    plt.bar(x + width / 2, post, width=width, label="Post-mitigation")
    plt.xticks(x, circuit_types)
    plt.title("Mean Fidelity Pre vs Post Error Mitigation")
    plt.xlabel("Circuit type")
    plt.ylabel("Mean fidelity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# =========================
# Main
# =========================

def parse_bool_csv(raw: str) -> List[bool]:
    values = []
    for item in raw.split(","):
        normalized = item.strip().lower()
        if normalized in {"true", "t", "1", "yes"}:
            values.append(True)
        elif normalized in {"false", "f", "0", "no"}:
            values.append(False)
        elif normalized:
            raise ValueError(f"Unsupported mitigation mode value: {item}")
    if not values:
        raise ValueError("At least one mitigation mode must be provided.")
    return values


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run the Mermin-Peres contextuality experiment.")
    parser.add_argument("--shots", type=int, default=int(os.getenv("MP_SHOTS", DEFAULT_SHOTS)))
    parser.add_argument(
        "--zne-scale-factors",
        default=os.getenv(
            "MP_ZNE_SCALE_FACTORS",
            ",".join(str(v) for v in DEFAULT_ZNE_SCALE_FACTORS),
        ),
    )
    parser.add_argument(
        "--noise-levels",
        default=os.getenv("MP_NOISE_LEVELS", ",".join(DEFAULT_NOISE_LEVEL_ORDER)),
    )
    parser.add_argument(
        "--circuit-types",
        default=os.getenv("MP_CIRCUIT_TYPES", ",".join(DEFAULT_CIRCUIT_TYPES)),
    )
    parser.add_argument(
        "--mitigation-modes",
        default=os.getenv("MP_MITIGATION_MODES", "false,true"),
        help="Comma-separated booleans, e.g. false,true",
    )
    parser.add_argument(
        "--contexts",
        default=os.getenv("MP_CONTEXTS", ",".join(QUERY_CONFIGS.keys())),
    )
    parser.add_argument(
        "--outdir",
        default=os.getenv("MP_OUTDIR", str(OUTDIR)),
    )
    parser.add_argument("--smoke-test", action="store_true", default=os.getenv("MP_SMOKE_TEST", "").lower() in {"1", "true", "yes"})
    parser.add_argument("--use-real-backend", action="store_true", default=USE_REAL_BACKEND)
    parser.add_argument("--use-backend-mimic", action="store_true", default=USE_BACKEND_MIMIC)
    parser.add_argument("--backend-name", default=os.getenv("MP_BACKEND_NAME", IBM_BACKEND_NAME))
    args = parser.parse_args()

    zne_scale_factors = [int(v.strip()) for v in args.zne_scale_factors.split(",") if v.strip()]
    noise_levels = [v.strip() for v in args.noise_levels.split(",") if v.strip()]
    circuit_types = [v.strip() for v in args.circuit_types.split(",") if v.strip()]
    contexts = [v.strip() for v in args.contexts.split(",") if v.strip()]
    mitigation_modes = parse_bool_csv(args.mitigation_modes)

    if args.smoke_test:
        if not noise_levels:
            noise_levels = ["low"]
        else:
            noise_levels = [noise_levels[0]]
        args.shots = min(args.shots, 256)
        zne_scale_factors = [1, 3]

    if args.use_real_backend and noise_levels == DEFAULT_NOISE_LEVEL_ORDER:
        noise_levels = [REAL_BACKEND_NOISE_LABEL]

    invalid_noise = [
        v for v in noise_levels
        if v not in NOISE_LEVELS and not (args.use_real_backend and v == REAL_BACKEND_NOISE_LABEL)
    ]
    invalid_circuits = [v for v in circuit_types if v not in DEFAULT_CIRCUIT_TYPES]
    invalid_contexts = [v for v in contexts if v not in QUERY_CONFIGS]
    if invalid_noise:
        raise ValueError(f"Unsupported noise levels: {invalid_noise}")
    if invalid_circuits:
        raise ValueError(f"Unsupported circuit types: {invalid_circuits}")
    if invalid_contexts:
        raise ValueError(f"Unsupported contexts: {invalid_contexts}")
    if args.use_real_backend and args.use_backend_mimic:
        raise ValueError("Choose either --use-real-backend or --use-backend-mimic, not both.")
    if any(sf < 1 or sf % 2 == 0 for sf in zne_scale_factors):
        raise ValueError("ZNE scale factors must be positive odd integers.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    return ExperimentConfig(
        shots=args.shots,
        zne_scale_factors=zne_scale_factors,
        noise_levels=noise_levels,
        circuit_types=circuit_types,
        mitigation_modes=mitigation_modes,
        contexts=contexts,
        outdir=outdir,
        use_real_backend=args.use_real_backend,
        use_backend_mimic=args.use_backend_mimic,
        backend_name=args.backend_name,
        smoke_test=args.smoke_test,
    )


def main() -> None:
    set_seeds(SEED)
    config = parse_args()
    print(
        f"[startup] real_backend={config.use_real_backend} backend_mimic={config.use_backend_mimic} "
        f"backend_name={config.backend_name} shots={config.shots} smoke_test={config.smoke_test}",
        flush=True,
    )

    contextual = build_contextuality_preserving_prep()
    contextual.name = "contextuality-preserving"

    baseline = build_baseline_prep()
    baseline.name = "baseline"

    prep_map = {
        "contextuality-preserving": contextual,
        "baseline": baseline,
    }

    records: List[RunRecord] = []
    context_records: List[ContextRecord] = []
    execution_manifest = {
        "seed": SEED,
        "shots": config.shots,
        "zne_scale_factors": config.zne_scale_factors,
        "noise_levels": config.noise_levels,
        "circuit_types": config.circuit_types,
        "mitigation_modes": config.mitigation_modes,
        "contexts": config.contexts,
        "backend_name": config.backend_name,
        "use_real_backend": config.use_real_backend,
        "use_backend_mimic": config.use_backend_mimic,
        "smoke_test": config.smoke_test,
        "generated_at_unix": time.time(),
    }
    save_json(execution_manifest, config.outdir / "run_manifest.json")

    total_runs = len(config.noise_levels) * len(config.mitigation_modes) * len(config.circuit_types)
    current_run = 0
    for noise_level in config.noise_levels:
        for mitigated in config.mitigation_modes:
            for circuit_type in config.circuit_types:
                current_run += 1
                print(
                    f"[batch] {current_run}/{total_runs} circuit={circuit_type} "
                    f"noise={noise_level} mitigated={mitigated}",
                    flush=True,
                )
                record, per_context = run_family(
                    config=config,
                    circuit_type=circuit_type,
                    prep_circuit=prep_map[circuit_type],
                    noise_level=noise_level,
                    mitigated=mitigated,
                )
                records.append(record)
                context_records.extend(per_context)

                partial_summary = pd.DataFrame([asdict(r) for r in records])
                save_dataframe(partial_summary, config.outdir / "summary_results.partial.csv")
                partial_context = pd.DataFrame([asdict(r) for r in context_records])
                save_dataframe(partial_context, config.outdir / "context_results.partial.csv")

    df = pd.DataFrame([asdict(r) for r in records])
    context_df = pd.DataFrame([asdict(r) for r in context_records])
    validate_probability_distributions(context_df)

    save_dataframe(df, config.outdir / "summary_results.csv")
    save_dataframe(context_df, config.outdir / "context_results.csv")
    save_dataframe(context_df, config.outdir / "query_results.csv")

    # Save a rounded version too
    df_rounded = df.copy()
    for col in ["fidelity", "trace_distance", "contextuality_violation", "win_probability"]:
        df_rounded[col] = df_rounded[col].round(6)
    save_dataframe(df_rounded, config.outdir / "summary_results_rounded.csv")

    summary_tables = build_interpretable_summary_tables(df, context_df)
    save_dataframe(summary_tables["paper_summary"], config.outdir / "table_paper_summary.csv")
    summary_tables["fidelity_table"].to_csv(config.outdir / "table_fidelity_by_noise.csv")
    summary_tables["win_rate_table"].to_csv(config.outdir / "table_win_rate_by_noise.csv")
    summary_tables["contextuality_table"].to_csv(config.outdir / "table_contextuality_by_noise.csv")
    save_dataframe(summary_tables["preserving_advantage"], config.outdir / "table_preserving_advantage.csv")
    save_dataframe(summary_tables["mitigation_gain"], config.outdir / "table_mitigation_gain.csv")
    save_dataframe(summary_tables["weakest_queries"], config.outdir / "table_weakest_queries.csv")
    write_text_report(df, context_df, config.outdir / "results_report.txt")
    workbook_saved, workbook_message = save_excel_workbook(
        config.outdir / "results_workbook.xlsx",
        summary_df=df,
        query_df=context_df,
        summary_tables=summary_tables,
        manifest=execution_manifest,
    )

    # Plots similar to the paper
    plot_fidelity_by_noise(df, config.outdir / "figure_fidelity_vs_noise.png")
    plot_win_probability_by_noise(df, config.outdir / "figure_win_rate_vs_noise.png")
    plot_contextuality_by_noise(df, config.outdir / "figure_contextuality_vs_noise.png")
    plot_contextuality_vs_fidelity(df, config.outdir / "figure_contextuality_vs_fidelity.png")
    if set(config.mitigation_modes) >= {False, True}:
        plot_pre_post_mitigation(df, config.outdir / "figure_pre_post_mitigation.png")
        plot_query_heatmaps(context_df, config.outdir)

    print("\n=== Finished ===")
    print(df_rounded.to_string(index=False))
    print(workbook_message)
    print(f"\nSaved outputs to: {config.outdir.resolve()}")


if __name__ == "__main__":
    main()

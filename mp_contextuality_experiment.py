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

# Optional IBM Runtime imports.
# Only needed if USE_REAL_BACKEND or USE_BACKEND_MIMIC is enabled.
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    IBM_RUNTIME_AVAILABLE = True
except Exception:
    IBM_RUNTIME_AVAILABLE = False


# =========================
# Configuration
# =========================

SEED = 7
OUTDIR = Path("mp_contextuality_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Backends:
USE_REAL_BACKEND = False
USE_BACKEND_MIMIC = False
IBM_BACKEND_NAME = "ibm_brisbane"

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
DEFAULT_CIRCUIT_TYPES = ["contextuality-preserving", "baseline"]
DEFAULT_MITIGATION_MODES = [False, True]


# =========================
# Utilities
# =========================

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_runtime_service() -> "QiskitRuntimeService":
    if not IBM_RUNTIME_AVAILABLE:
        raise RuntimeError(
            "qiskit-ibm-runtime is not installed. Install it or disable real-backend options."
        )
    return QiskitRuntimeService()


def get_reference_backend():
    """Optional backend used either directly or to mimic a noise profile."""
    service = ensure_runtime_service()
    backend = service.backend(IBM_BACKEND_NAME)
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
        backend = get_reference_backend()
        sim = AerSimulator.from_backend(backend)
        return sim, backend
    else:
        sim = AerSimulator(
            noise_model=build_local_noise_model(level),
            seed_simulator=SEED,
        )
        return sim, None


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
    simulator, ref_backend = build_simulator(noise_level)

    backend_name = config.backend_name if config.use_backend_mimic else "local_aer_noise_model"
    raw_context_scale_dists: Dict[str, Dict[int, Dict[str, float]]] = {}

    # Optional real backend path.
    if config.use_real_backend:
        backend = get_reference_backend()
        backend_name = backend.name
        context_dists = {}

        # Real backend run without mitigation or with manual ZNE via repeated jobs.
        for ctx in config.contexts:
            print(f"[run] circuit={circuit_type} noise={noise_level} mitigated={mitigated} query={ctx}", flush=True)
            base_meas = build_query_circuit(prep_circuit, ctx, include_measurements=True)
            if mitigated:
                raw_context_scale_dists[ctx] = {}
                for sf in config.zne_scale_factors:
                    print(
                        f"[run] circuit={circuit_type} noise={noise_level} mitigated={mitigated} "
                        f"query={ctx} scale_factor={sf}",
                        flush=True,
                    )
                    folded = fold_global(base_meas, sf)
                    compiled = compile_circuit(folded, backend)
                    job = backend.run(compiled, shots=config.shots)
                    result = job.result()
                    counts = result.get_counts()
                    dist = outcome_distribution_from_counts(counts, config.shots)
                    raw_context_scale_dists[ctx][sf] = dist
                outcome_labels = tuple(format(index, "04b") for index in range(16))
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
                compiled = compile_circuit(base_meas, backend)
                job = backend.run(compiled, shots=config.shots)
                result = job.result()
                counts = result.get_counts()
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


def plot_fidelity_by_noise(df: pd.DataFrame, outpath: Path) -> None:
    pivot = (
        df[df["mitigated"] == False]
        .pivot(index="noise_level", columns="circuit_type", values="fidelity")
        .reindex(["low", "medium", "high"])
    )

    plt.figure(figsize=(7, 5))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], marker="o", label=col)
    plt.title("Mean Fidelity as a Function of Noise Level")
    plt.xlabel("Noise level")
    plt.ylabel("Mean fidelity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_contextuality_by_noise(df: pd.DataFrame, outpath: Path) -> None:
    pivot = (
        df[df["mitigated"] == False]
        .pivot(index="noise_level", columns="circuit_type", values="contextuality_violation")
        .reindex(["low", "medium", "high"])
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

    invalid_noise = [v for v in noise_levels if v not in NOISE_LEVELS]
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

    # Save a rounded version too
    df_rounded = df.copy()
    for col in ["fidelity", "trace_distance", "contextuality_violation", "win_probability"]:
        df_rounded[col] = df_rounded[col].round(6)
    save_dataframe(df_rounded, config.outdir / "summary_results_rounded.csv")

    # Plots similar to the paper
    if set(config.noise_levels) >= {"low", "medium", "high"} and len(config.noise_levels) >= 3:
        plot_fidelity_by_noise(df, config.outdir / "figure_fidelity_vs_noise.png")
        plot_contextuality_by_noise(df, config.outdir / "figure_contextuality_vs_noise.png")
    plot_contextuality_vs_fidelity(df, config.outdir / "figure_contextuality_vs_fidelity.png")
    if set(config.mitigation_modes) >= {False, True}:
        plot_pre_post_mitigation(df, config.outdir / "figure_pre_post_mitigation.png")

    print("\n=== Finished ===")
    print(df_rounded.to_string(index=False))
    print(f"\nSaved outputs to: {config.outdir.resolve()}")


if __name__ == "__main__":
    main()

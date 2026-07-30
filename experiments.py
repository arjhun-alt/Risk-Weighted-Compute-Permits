from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from mesa import Agent, Model


Coupling = Literal["independent", "lower_frechet"]
Channel = Literal["evaluation", "compute", "combined"]
Behaviour = Literal[
    "exact",
    "logit_4",
    "logit_1_5",
    "heterogeneous",
]

IMPLEMENTATION_SEEDS = tuple(range(11, 17))
BEHAVIOURAL_SEEDS = tuple(range(42, 50))
N_DEVELOPERS = 400
NUMERICAL_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ImplementationParameters:
    alpha: float = 4.0
    compute: float = 1.0
    price: float = 1.0
    penalty_slope: float = 0.1
    penalty_curvature: float = 0.05
    evaluation_cost: float = 0.01
    compute_cost: float = 0.01
    effort_upper: float = 10.0
    effort_points: int = 11
    hidden_compute_points: int = 11
    deviation_points: int = 31
    developers: int = N_DEVELOPERS


@dataclass(frozen=True)
class BehaviouralParameters:
    alpha: float = 2.0
    compute: float = 1.0
    price: float = 1.0
    audit_probability: float = 0.7
    penalty_slope: float = 2.0
    penalty_curvature: float = 1.0
    heterogeneity_half_width: float = 0.15
    developers: int = N_DEVELOPERS


@dataclass(frozen=True)
class Action:
    deviation: float
    hidden_compute: float
    evaluation_effort: float
    compute_effort: float
    evaluation_hiddenness: float
    compute_hiddenness: float
    probabilities: tuple[float, float, float, float]
    payoff: float


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_probability(name: str, value: float) -> None:
    _require_finite(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _require_nonnegative(name: str, value: float) -> None:
    _require_finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")


def evaluation_success(effort: float) -> float:
    _require_nonnegative("effort", effort)
    return 1.0 - math.exp(-3.0 * effort)


def evaluation_detection(effort: float) -> float:
    _require_nonnegative("effort", effort)
    return 0.05 * math.exp(-0.5 * effort)


def evaluation_hiddenness(effort: float) -> float:
    return evaluation_success(effort) * (1.0 - evaluation_detection(effort))


def compute_success(hidden_compute: float, effort: float, compute: float = 1.0) -> float:
    _require_nonnegative("hidden_compute", hidden_compute)
    _require_nonnegative("effort", effort)
    if compute <= 0.0 or hidden_compute > compute:
        raise ValueError("compute must be positive and hidden_compute must not exceed it")
    if hidden_compute == 0.0 or effort == 0.0:
        return 0.0
    return (1.0 - math.exp(-3.0 * effort)) * (1.0 - hidden_compute / compute)


def compute_detection(hidden_compute: float, effort: float, compute: float = 1.0) -> float:
    _require_nonnegative("hidden_compute", hidden_compute)
    _require_nonnegative("effort", effort)
    if compute <= 0.0 or hidden_compute > compute:
        raise ValueError("compute must be positive and hidden_compute must not exceed it")
    if hidden_compute == 0.0:
        return 0.0
    return 0.1 * (hidden_compute / compute) / (1.0 + effort)


def compute_hiddenness(hidden_compute: float, effort: float, compute: float = 1.0) -> float:
    if hidden_compute == 0.0:
        _require_nonnegative("effort", effort)
        return 0.0
    return compute_success(hidden_compute, effort, compute) * (
        1.0 - compute_detection(hidden_compute, effort, compute)
    )


def joint_hiddenness(
    evaluation: float,
    compute: float,
    coupling: Coupling,
) -> float:
    _require_probability("evaluation hiddenness", evaluation)
    _require_probability("compute hiddenness", compute)
    if coupling == "independent":
        return evaluation * compute
    if coupling == "lower_frechet":
        return max(0.0, evaluation + compute - 1.0)
    raise ValueError(f"unknown coupling: {coupling}")


def four_state_probabilities(
    evaluation: float,
    compute: float,
    coupling: Coupling,
) -> tuple[float, float, float, float]:
    joint = joint_hiddenness(evaluation, compute, coupling)
    probabilities = (
        1.0 - evaluation - compute + joint,
        evaluation - joint,
        compute - joint,
        joint,
    )
    if not all(math.isfinite(value) for value in probabilities):
        raise ValueError("four-state probabilities must be finite")
    if not all(-NUMERICAL_TOLERANCE <= value <= 1.0 + NUMERICAL_TOLERANCE for value in probabilities):
        raise ValueError("four-state probabilities must lie in [0, 1]")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=NUMERICAL_TOLERANCE):
        raise ValueError("four-state probabilities must sum to one")
    return tuple(0.0 if abs(value) < NUMERICAL_TOLERANCE else value for value in probabilities)


def state_gaps(
    alpha: float,
    compute: float,
    hidden_compute: float,
) -> tuple[float, float, float, float]:
    if alpha < 1.0 or compute <= 0.0 or not 0.0 <= hidden_compute <= compute:
        raise ValueError("invalid liability parameters")
    return (
        0.0,
        (alpha - 1.0) * compute,
        alpha * hidden_compute,
        (alpha - 1.0) * compute + hidden_compute,
    )


def expected_hidden_liability(
    alpha: float,
    compute: float,
    hidden_compute: float,
    evaluation: float,
    compute_hidden: float,
    coupling: Coupling,
) -> float:
    joint = joint_hiddenness(evaluation, compute_hidden, coupling)
    return (
        (alpha - 1.0) * compute * evaluation
        + alpha * hidden_compute * compute_hidden
        - (alpha - 1.0) * hidden_compute * joint
    )


def detected_shortfall(deviation: float, hidden_liability_gap: float) -> float:
    _require_nonnegative("deviation", deviation)
    _require_nonnegative("hidden_liability_gap", hidden_liability_gap)
    return max(deviation - hidden_liability_gap, 0.0)


def quadratic_penalty(shortfall: np.ndarray | float, slope: float, curvature: float):
    return slope * shortfall + curvature * shortfall * shortfall


def developer_payoff(
    deviation: np.ndarray | float,
    hidden_compute: float,
    evaluation_effort: float,
    compute_effort: float,
    audit_probability: float,
    probabilities: tuple[float, float, float, float],
    params: ImplementationParameters,
    evaluation_active: bool,
    compute_active: bool,
):
    gaps = state_gaps(params.alpha, params.compute, hidden_compute)
    expected_penalty = 0.0
    for probability, gap in zip(probabilities, gaps):
        shortfall = np.maximum(np.asarray(deviation) - gap, 0.0)
        expected_penalty += probability * quadratic_penalty(
            shortfall,
            params.penalty_slope,
            params.penalty_curvature,
        )
    private_cost = 0.0
    if evaluation_active:
        private_cost += params.evaluation_cost * evaluation_effort**2
    if compute_active:
        private_cost += params.compute_cost * (
            compute_effort**2 + (hidden_compute / params.compute) ** 2
        )
    return params.price * np.asarray(deviation) - private_cost - audit_probability * expected_penalty


def grid_best_response(
    channel: Channel,
    coupling: Coupling,
    audit_probability: float,
    params: ImplementationParameters | None = None,
) -> Action:
    params = params or ImplementationParameters()
    _require_probability("audit_probability", audit_probability)
    evaluation_active = channel in ("evaluation", "combined")
    compute_active = channel in ("compute", "combined")
    if channel not in ("evaluation", "compute", "combined"):
        raise ValueError(f"unknown channel: {channel}")

    effort_grid = np.linspace(0.0, params.effort_upper, params.effort_points)
    hidden_compute_grid = np.linspace(
        0.0,
        params.compute,
        params.hidden_compute_points,
    )
    deviation_grid = np.linspace(
        0.0,
        params.alpha * params.compute,
        params.deviation_points,
    )
    evaluation_grid = effort_grid if evaluation_active else np.array([0.0])
    compute_grid = effort_grid if compute_active else np.array([0.0])
    hidden_grid = hidden_compute_grid if compute_active else np.array([0.0])

    best = Action(
        deviation=0.0,
        hidden_compute=0.0,
        evaluation_effort=0.0,
        compute_effort=0.0,
        evaluation_hiddenness=0.0,
        compute_hiddenness=0.0,
        probabilities=(1.0, 0.0, 0.0, 0.0),
        payoff=0.0,
    )
    for hidden_compute in hidden_grid:
        ell = float(hidden_compute)
        for evaluation_effort in evaluation_grid:
            x_e = float(evaluation_effort)
            hidden_e = evaluation_hiddenness(x_e) if evaluation_active else 0.0
            for compute_effort in compute_grid:
                x_c = float(compute_effort)
                hidden_c = compute_hiddenness(ell, x_c, params.compute) if compute_active else 0.0
                probabilities = four_state_probabilities(hidden_e, hidden_c, coupling)
                payoffs = developer_payoff(
                    deviation_grid,
                    ell,
                    x_e,
                    x_c,
                    audit_probability,
                    probabilities,
                    params,
                    evaluation_active,
                    compute_active,
                )
                index = int(np.argmax(payoffs))
                payoff = float(payoffs[index])
                if payoff > best.payoff + NUMERICAL_TOLERANCE:
                    best = Action(
                        deviation=float(deviation_grid[index]),
                        hidden_compute=ell,
                        evaluation_effort=x_e,
                        compute_effort=x_c,
                        evaluation_hiddenness=hidden_e,
                        compute_hiddenness=hidden_c,
                        probabilities=probabilities,
                        payoff=payoff,
                    )
    return best


class VerificationAgent(Agent):
    def __init__(self, model: "VerificationModel", action: Action):
        super().__init__(model)
        self.action = action
        self.audited = False
        self.realized_gap = math.nan

    def step(self) -> None:
        generator = self.model.generator
        self.audited = bool(generator.random() < self.model.audit_probability)
        if not self.audited:
            return
        draw = generator.random()
        cumulative = 0.0
        state = 3
        for index, probability in enumerate(self.action.probabilities):
            cumulative += probability
            if draw < cumulative:
                state = index
                break
        self.realized_gap = state_gaps(
            self.model.params.alpha,
            self.model.params.compute,
            self.action.hidden_compute,
        )[state]


class VerificationModel(Model):
    def __init__(
        self,
        action: Action,
        audit_probability: float,
        seed: int,
        params: ImplementationParameters,
    ):
        super().__init__(rng=seed)
        self.generator = np.random.default_rng(seed)
        self.audit_probability = audit_probability
        self.params = params
        self.developers: list[VerificationAgent] = []
        for _ in range(params.developers):
            self.generator.random()
            self.developers.append(VerificationAgent(self, action))

    def run(self) -> float:
        for developer in self.developers:
            developer.step()
        gaps = [developer.realized_gap for developer in self.developers if developer.audited]
        if not gaps:
            raise RuntimeError("the seeded run produced no audits")
        return float(np.mean(gaps))


def _implementation_configurations() -> tuple[tuple[str, Channel, Coupling], ...]:
    return (
        ("Evaluation only", "evaluation", "independent"),
        ("Compute only", "compute", "independent"),
        ("Combined: independent", "combined", "independent"),
        ("Combined: lower Frechet", "combined", "lower_frechet"),
    )


def run_implementation_verification(
    output_directory: str | Path = "results",
) -> dict[str, object]:
    started = time.perf_counter()
    params = ImplementationParameters()
    rows: list[dict[str, object]] = []
    for audit_probability in (0.1, 0.2, 0.3):
        for label, channel, coupling in _implementation_configurations():
            action = grid_best_response(channel, coupling, audit_probability, params)
            closed_form = expected_hidden_liability(
                params.alpha,
                params.compute,
                action.hidden_compute,
                action.evaluation_hiddenness,
                action.compute_hiddenness,
                coupling,
            )
            sampled = [
                VerificationModel(action, audit_probability, seed, params).run()
                for seed in IMPLEMENTATION_SEEDS
            ]
            simulated_mean = float(np.mean(sampled))
            standard_error = float(np.std(sampled, ddof=0) / math.sqrt(len(sampled)))
            if abs(closed_form) <= NUMERICAL_TOLERANCE:
                relative_difference = abs(simulated_mean - closed_form)
            else:
                relative_difference = abs(simulated_mean - closed_form) / abs(closed_form)
            rows.append(
                {
                    "configuration": label,
                    "channel": channel,
                    "coupling": coupling,
                    "rho": audit_probability,
                    "closed_form_mean": closed_form,
                    "simulated_mean": simulated_mean,
                    "standard_error": standard_error,
                    "relative_difference": relative_difference,
                    "n_developers": params.developers,
                    "n_seeds": len(IMPLEMENTATION_SEEDS),
                    "seed_start": IMPLEMENTATION_SEEDS[0],
                    "seed_end": IMPLEMENTATION_SEEDS[-1],
                    "deviation": action.deviation,
                    "hidden_compute": action.hidden_compute,
                    "evaluation_effort": action.evaluation_effort,
                    "compute_effort": action.compute_effort,
                }
            )
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path / "implementation_verification.csv", rows, precision=10)
    return {
        "rows": rows,
        "runtime_seconds": time.perf_counter() - started,
        "max_relative_difference": max(float(row["relative_difference"]) for row in rows),
    }


def binary_gain(hiddenness: float, params: BehaviouralParameters | None = None) -> float:
    params = params or BehaviouralParameters()
    _require_probability("hiddenness", hiddenness)
    deviation = (params.alpha - 1.0) * params.compute
    penalty = quadratic_penalty(
        deviation,
        params.penalty_slope,
        params.penalty_curvature,
    )
    return params.price * deviation - params.audit_probability * (1.0 - hiddenness) * penalty


def deviation_probability(
    hiddenness: float,
    response: float | None,
    params: BehaviouralParameters | None = None,
) -> float:
    gain = binary_gain(hiddenness, params)
    if response is None:
        return 1.0 if gain > 0.0 else 0.0
    _require_nonnegative("response", response)
    scaled = response * gain
    if scaled >= 0.0:
        return 1.0 / (1.0 + math.exp(-scaled))
    exponential = math.exp(scaled)
    return exponential / (1.0 + exponential)


def behavioural_thresholds(
    params: BehaviouralParameters | None = None,
) -> tuple[float, float]:
    params = params or BehaviouralParameters()
    deviation = (params.alpha - 1.0) * params.compute
    penalty_at_deviation = quadratic_penalty(
        deviation,
        params.penalty_slope,
        params.penalty_curvature,
    )
    binary = 1.0 - params.price * deviation / (
        params.audit_probability * penalty_at_deviation
    )
    marginal = 1.0 - params.price / (
        params.audit_probability * params.penalty_slope
    )
    return float(binary), float(marginal)


class BehaviouralAgent(Agent):
    def __init__(
        self,
        model: "BehaviouralModel",
        hiddenness: float,
        response: float | None,
    ):
        super().__init__(model)
        self.hiddenness = hiddenness
        self.response = response
        self.compliant = True

    def step(self) -> None:
        probability = deviation_probability(
            self.hiddenness,
            self.response,
            self.model.params,
        )
        if self.response is None:
            self.compliant = probability == 0.0
        else:
            self.compliant = not bool(self.model.generator.random() < probability)


class BehaviouralModel(Model):
    def __init__(
        self,
        hiddenness: float,
        behaviour: Behaviour,
        seed: int,
        params: BehaviouralParameters,
    ):
        super().__init__(rng=seed)
        self.generator = np.random.default_rng(seed)
        self.params = params
        if behaviour == "logit_4":
            response = 4.0
        elif behaviour == "logit_1_5":
            response = 1.5
        elif behaviour in ("exact", "heterogeneous"):
            response = None
        else:
            raise ValueError(f"unknown behaviour: {behaviour}")

        self.developers: list[BehaviouralAgent] = []
        for _ in range(params.developers):
            self.generator.random()
            individual_hiddenness = hiddenness
            if behaviour == "heterogeneous":
                draw = self.generator.uniform(-1.0, 1.0)
                individual_hiddenness = float(
                    np.clip(
                        hiddenness + params.heterogeneity_half_width * draw,
                        0.0,
                        1.0,
                    )
                )
            self.developers.append(
                BehaviouralAgent(self, individual_hiddenness, response)
            )

    def run(self) -> float:
        for developer in self.developers:
            developer.step()
        return sum(developer.compliant for developer in self.developers) / len(self.developers)


def run_behavioural_extension(
    output_directory: str | Path = "results",
) -> dict[str, object]:
    started = time.perf_counter()
    params = BehaviouralParameters()
    behaviours: tuple[tuple[str, Behaviour], ...] = (
        ("exact_choice", "exact"),
        ("logit_lambda_4", "logit_4"),
        ("logit_lambda_1_5", "logit_1_5"),
        ("heterogeneous_hiddenness", "heterogeneous"),
    )
    rows: list[dict[str, object]] = []
    for hiddenness in np.linspace(0.0, 1.0, 21):
        row: dict[str, object] = {"h": float(hiddenness)}
        for column, behaviour in behaviours:
            compliance = [
                BehaviouralModel(float(hiddenness), behaviour, seed, params).run()
                for seed in BEHAVIOURAL_SEEDS
            ]
            row[column] = float(np.mean(compliance))
        row.update(
            {
                "n_developers": params.developers,
                "n_seeds": len(BEHAVIOURAL_SEEDS),
                "seed_start": BEHAVIOURAL_SEEDS[0],
                "seed_end": BEHAVIOURAL_SEEDS[-1],
            }
        )
        rows.append(row)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path / "behavioural_extension.csv", rows, precision=4)
    binary, marginal = behavioural_thresholds(params)
    return {
        "rows": rows,
        "runtime_seconds": time.perf_counter() - started,
        "binary_break_even": binary,
        "marginal_threshold": marginal,
    }


def plot_implementation_verification(result: dict[str, object]):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.4, 4.6), constrained_layout=True)
    styles = {
        "Evaluation only": ("o", "#3D6B4A", True),
        "Compute only": ("^", "#3D6B4A", False),
        "Combined: independent": ("s", "#B85F65", True),
        "Combined: lower Frechet": ("D", "#B85F65", False),
    }
    rows = result["rows"]
    for label, (marker, color, filled) in styles.items():
        selected = [row for row in rows if row["configuration"] == label]
        axis.errorbar(
            [row["closed_form_mean"] for row in selected],
            [row["simulated_mean"] for row in selected],
            yerr=[row["standard_error"] for row in selected],
            fmt=marker,
            color=color,
            markerfacecolor=color if filled else "white",
            capsize=3,
            label=label,
        )
    axis.plot([0.0, 3.5], [0.0, 3.5], "--", color="#222222", linewidth=1)
    axis.set(
        xlim=(0.0, 3.5),
        ylim=(0.0, 3.5),
        xlabel="Closed-form expected hidden liability",
        ylabel="Simulated mean hidden liability",
    )
    axis.set_aspect("equal")
    axis.grid(color="#D8D8D3", linewidth=0.6)
    axis.legend(frameon=False, fontsize=8)
    return figure


def plot_behavioural_extension(result: dict[str, object]):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
    rows = result["rows"]
    hiddenness = [row["h"] for row in rows]
    binary = float(result["binary_break_even"])
    marginal = float(result["marginal_threshold"])
    axis.axvspan(marginal, binary, color="#D8D8D3", alpha=0.55)
    axis.axvline(marginal, color="gray", linestyle=":", linewidth=1)
    axis.axvline(binary, color="gray", linestyle=":", linewidth=1)
    axis.plot(hiddenness, [row["exact_choice"] for row in rows], color="#222222", label="Exact choice")
    axis.plot(
        hiddenness,
        [row["logit_lambda_4"] for row in rows],
        "--",
        color="#B85F65",
        label=r"Logit: $\lambda=4$",
    )
    axis.plot(
        hiddenness,
        [row["logit_lambda_1_5"] for row in rows],
        ":",
        color="#B85F65",
        label=r"Logit: $\lambda=1.5$",
    )
    axis.plot(
        hiddenness,
        [row["heterogeneous_hiddenness"] for row in rows],
        "-.",
        color="#3D6B4A",
        label="Heterogeneous hiddenness",
    )
    axis.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="Evaluation-channel hiddenness, h",
        ylabel="Compliant fraction",
    )
    axis.grid(axis="y", color="#D8D8D3", linewidth=0.6)
    axis.legend(frameon=False, fontsize=8)
    return figure


def validate_equations() -> None:
    assert evaluation_success(0.0) == 0.0
    assert compute_success(0.0, 1.0) == 0.0
    assert compute_success(0.5, 0.0) == 0.0
    assert compute_hiddenness(0.0, 1.0) == 0.0
    for evaluation in np.linspace(0.0, 1.0, 11):
        for compute in np.linspace(0.0, 1.0, 11):
            for coupling in ("independent", "lower_frechet"):
                probabilities = four_state_probabilities(
                    float(evaluation),
                    float(compute),
                    coupling,
                )
                assert all(value >= 0.0 for value in probabilities)
                assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-12)
                hidden_compute = 0.4
                gaps = state_gaps(4.0, 1.0, hidden_compute)
                weighted = sum(value * gap for value, gap in zip(probabilities, gaps))
                reduced = expected_hidden_liability(
                    4.0,
                    1.0,
                    hidden_compute,
                    float(evaluation),
                    float(compute),
                    coupling,
                )
                assert math.isclose(weighted, reduced, abs_tol=1e-12)
    assert state_gaps(4.0, 1.0, 0.4) == (0.0, 3.0, 1.6, 3.4)
    binary, marginal = behavioural_thresholds()
    assert math.isclose(binary, 0.5238095238095238, abs_tol=1e-14)
    assert math.isclose(marginal, 0.2857142857142857, abs_tol=1e-14)
    assert binary_gain(binary) == 0.0
    assert deviation_probability(binary, None) == 0.0
    action = grid_best_response("evaluation", "independent", 0.1)
    first = VerificationModel(action, 0.1, 11, ImplementationParameters()).run()
    second = VerificationModel(action, 0.1, 11, ImplementationParameters()).run()
    assert first == second


def _format_value(value: object, precision: int) -> object:
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return value


def _write_csv(path: Path, rows: list[dict[str, object]], precision: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(value, precision) for key, value in row.items()})


def _write_summary(
    implementation: dict[str, object],
    behavioural: dict[str, object],
    output_directory: Path,
) -> None:
    summary = {
        "study_scope": [
            "Implementation Verification",
            "Behavioural Extension",
        ],
        "formal_model_computation": True,
        "implementation_verification": {
            "points": len(implementation["rows"]),
            "developers_per_run": N_DEVELOPERS,
            "seeds": list(IMPLEMENTATION_SEEDS),
            "max_relative_difference": implementation["max_relative_difference"],
            "runtime_seconds": implementation["runtime_seconds"],
        },
        "behavioural_extension": {
            "points": len(behavioural["rows"]),
            "developers_per_run": N_DEVELOPERS,
            "seeds": list(BEHAVIOURAL_SEEDS),
            "binary_break_even": behavioural["binary_break_even"],
            "marginal_threshold": behavioural["marginal_threshold"],
            "runtime_seconds": behavioural["runtime_seconds"],
        },
    }
    summary_path = output_directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def reproduce_all(output_directory: str | Path = "results") -> dict[str, object]:
    validate_equations()
    output_path = Path(output_directory)
    implementation = run_implementation_verification(output_path)
    behavioural = run_behavioural_extension(output_path)
    _write_summary(implementation, behavioural, output_path)
    return {
        "implementation_verification": implementation,
        "behavioural_extension": behavioural,
    }


if __name__ == "__main__":
    reproduced = reproduce_all()
    implementation = reproduced["implementation_verification"]
    behavioural = reproduced["behavioural_extension"]
    print(
        "Implementation Verification:",
        f"{len(implementation['rows'])} points,",
        f"maximum relative difference {100 * implementation['max_relative_difference']:.1f}%,",
        f"{implementation['runtime_seconds']:.2f} seconds",
    )
    print(
        "Behavioural Extension:",
        f"{len(behavioural['rows'])} points,",
        f"h*={behavioural['binary_break_even']:.6f},",
        f"h_marg={behavioural['marginal_threshold']:.6f},",
        f"{behavioural['runtime_seconds']:.2f} seconds",
    )

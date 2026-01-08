#!/usr/bin/env python3
"""Generate sampler test fixtures from k-diffusion formulas.

This script generates reference outputs that our Rust samplers can test against
to ensure k-diffusion compatibility.

Usage:
    pip install torch numpy
    python scripts/gen_sampler_fixtures.py

Outputs JSON fixtures to crates/burn-models-samplers/tests/fixtures/
"""

import json
import math
import numpy as np
from pathlib import Path

# Output directory
FIXTURES_DIR = Path(__file__).parent.parent / "crates/burn-models-samplers/tests/fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


def get_karras_sigmas(n_steps: int, sigma_min: float = 0.0292, sigma_max: float = 14.6146, rho: float = 7.0) -> list:
    """Generate Karras sigma schedule (matches k-diffusion exactly)."""
    ramp = np.linspace(0, 1, n_steps + 1)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    sigmas = sigmas.tolist()
    sigmas[-1] = 0.0  # Final sigma is exactly 0
    return sigmas


def generate_sigma_schedule_fixtures():
    """Generate sigma schedule test fixtures."""
    print("Generating sigma schedule fixtures...")

    sigma_min = 0.0292
    sigma_max = 14.6146
    rho = 7.0

    fixture = {
        "karras": {
            "5_steps": get_karras_sigmas(5, sigma_min, sigma_max, rho),
            "10_steps": get_karras_sigmas(10, sigma_min, sigma_max, rho),
            "20_steps": get_karras_sigmas(20, sigma_min, sigma_max, rho),
        },
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "rho": rho,
    }

    output_path = FIXTURES_DIR / "sigma_schedules.json"
    with open(output_path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"  Saved to {output_path}")


def generate_dpm_2m_fixtures():
    """Generate DPM++ 2M intermediate value fixtures."""
    print("Generating DPM++ 2M fixtures...")

    sigmas = get_karras_sigmas(5)
    test_cases = []

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Skip final step (sigma_next=0) since t_next would be infinity
        if sigma_next == 0:
            continue

        t = -math.log(sigma)
        t_next = -math.log(sigma_next)
        h = t_next - t

        case = {
            "name": f"step_{i}",
            "sigma": sigma,
            "sigma_next": sigma_next,
            "expected": {
                "t": t,
                "t_next": t_next,
                "h": h,
            },
        }

        if i > 0:
            old_sigma = sigmas[i - 1]
            t_prev = -math.log(old_sigma)
            h_prev = t - t_prev
            r = h_prev / h

            case["old_sigma"] = old_sigma
            case["expected"]["t_prev"] = t_prev
            case["expected"]["h_prev"] = h_prev
            case["expected"]["r"] = r

        test_cases.append(case)

    fixture = {
        "name": "dpm_2m_k_diffusion_reference",
        "description": "Reference values for DPM++ 2M using k-diffusion formulation",
        "formulas": {
            "first_order": "x_next = (sigma_next / sigma) * x + (1 - exp(-h)) * denoised",
            "second_order": "denoised_d = (1 + 1/(2r)) * denoised - (1/(2r)) * old_denoised; x_next = (sigma_next / sigma) * x + (1 - exp(-h)) * denoised_d",
            "where": {
                "t": "-log(sigma)",
                "h": "t_next - t",
                "r": "h_prev / h",
            },
        },
        "test_cases": test_cases,
        "coefficient_verification": {
            "description": "For r = h_prev / h, coeff = 1/(2r)",
            "examples": [
                {"r": 1.0, "coeff": 0.5, "d_weight": 1.5, "old_d_weight": -0.5},
                {"r": 0.5, "coeff": 1.0, "d_weight": 2.0, "old_d_weight": -1.0},
                {"r": 2.0, "coeff": 0.25, "d_weight": 1.25, "old_d_weight": -0.25},
            ],
        },
    }

    output_path = FIXTURES_DIR / "dpm_2m_reference.json"
    with open(output_path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"  Saved to {output_path}")


def generate_euler_ancestral_fixtures():
    """Generate Euler Ancestral intermediate value fixtures."""
    print("Generating Euler Ancestral fixtures...")

    sigmas = get_karras_sigmas(5)
    eta = 1.0
    s_noise = 1.0
    steps = []

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        if sigma_next == 0:
            steps.append({
                "step_index": i,
                "sigma": sigma,
                "sigma_next": sigma_next,
                "final_step": True,
            })
            continue

        # k-diffusion ancestral noise parameters
        sigma_up_unscaled = math.sqrt(
            sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2
        )
        sigma_up = min(eta * sigma_up_unscaled, sigma_next) * s_noise
        sigma_down = math.sqrt(sigma_next**2 - sigma_up**2)
        dt = sigma_down - sigma

        steps.append({
            "step_index": i,
            "sigma": sigma,
            "sigma_next": sigma_next,
            "expected": {
                "sigma_up": sigma_up,
                "sigma_down": sigma_down,
                "dt": dt,
            },
        })

    fixture = {
        "name": "euler_ancestral_k_diffusion_reference",
        "description": "Reference values for Euler Ancestral using k-diffusion formulation",
        "eta": eta,
        "s_noise": s_noise,
        "sigmas": sigmas,
        "steps": steps,
    }

    output_path = FIXTURES_DIR / "euler_ancestral_reference.json"
    with open(output_path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"  Saved to {output_path}")


def generate_dpm_sde_fixtures():
    """Generate DPM++ 2M SDE intermediate value fixtures."""
    print("Generating DPM++ 2M SDE fixtures...")

    sigmas = get_karras_sigmas(5)
    eta = 1.0
    s_noise = 1.0
    steps = []

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        if sigma_next == 0:
            steps.append({
                "step_index": i,
                "sigma": sigma,
                "sigma_next": sigma_next,
                "final_step": True,
            })
            continue

        t = -math.log(sigma)
        t_next = -math.log(sigma_next)
        h = t_next - t

        # SDE noise parameters
        exp_neg_2_eta_h = math.exp(-2 * eta * h)
        sigma_up = sigma_next * math.sqrt(max(0, 1 - exp_neg_2_eta_h)) * s_noise
        exp_neg_eta_h = math.exp(-eta * h)
        sigma_down = sigma_next * exp_neg_eta_h

        step = {
            "step_index": i,
            "sigma": sigma,
            "sigma_next": sigma_next,
            "expected": {
                "t": t,
                "t_next": t_next,
                "h": h,
                "sigma_up": sigma_up,
                "sigma_down": sigma_down,
            },
        }

        if i > 0:
            old_sigma = sigmas[i - 1]
            t_prev = -math.log(old_sigma)
            h_prev = t - t_prev
            r = h_prev / h
            step["old_sigma"] = old_sigma
            step["expected"]["t_prev"] = t_prev
            step["expected"]["h_prev"] = h_prev
            step["expected"]["r"] = r

        steps.append(step)

    fixture = {
        "name": "dpm_2m_sde_k_diffusion_reference",
        "description": "Reference values for DPM++ 2M SDE using k-diffusion formulation",
        "eta": eta,
        "s_noise": s_noise,
        "sigmas": sigmas,
        "steps": steps,
    }

    output_path = FIXTURES_DIR / "dpm_sde_reference.json"
    with open(output_path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"  Saved to {output_path}")


def main():
    print("Generating sampler test fixtures from k-diffusion formulas...")
    print(f"Output directory: {FIXTURES_DIR}\n")

    generate_sigma_schedule_fixtures()
    generate_dpm_2m_fixtures()
    generate_euler_ancestral_fixtures()
    generate_dpm_sde_fixtures()

    print("\nDone! Run Rust tests with: cargo test -p burn-models-samplers --test sampler_fixtures")


if __name__ == "__main__":
    main()

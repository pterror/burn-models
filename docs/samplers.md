# Samplers

Samplers control how noise is removed during the diffusion process. Different samplers have different speed/quality tradeoffs.

## Quick Reference

| Sampler | Steps | Speed | Quality | Stochastic | Best For |
|---------|-------|-------|---------|------------|----------|
| **DPM++ 2M** | 15-25 | Fast | Excellent | No | General use |
| **DPM++ 2M Karras** | 15-25 | Fast | Excellent | No | General use |
| **DPM++ SDE** | 20-30 | Medium | Excellent | Yes | Variety |
| **Euler** | 20-30 | Fast | Good | No | Speed |
| **Euler Ancestral** | 20-30 | Fast | Good | Yes | Creative variety |
| **DDIM** | 20-50 | Medium | Good | No | Consistency |
| **LCM** | 4-8 | Very Fast | Good | No | Speed priority |
| **UniPC** | 10-20 | Fast | Excellent | No | Low step counts |
| **DEIS** | 10-20 | Fast | Excellent | No | Low step counts |
| **Heun** | 20-30 | Slow | Excellent | No | Quality priority |

## Sampler Types

### Deterministic Samplers

These produce the same output given the same seed and parameters.

#### DPM++ 2M

The default choice for most use cases. Based on the DPM-Solver++ paper.

```rust
use burn_image_samplers::{DpmPlusPlusSampler, DpmConfig};

let config = DpmConfig {
    num_inference_steps: 20,
    use_karras_sigmas: true,  // Recommended
    ..Default::default()
};
let sampler = DpmPlusPlusSampler::new(config, &schedule);
```

#### DDIM

Classic denoising diffusion implicit model. Good baseline.

```rust
use burn_image_samplers::{DdimSampler, DdimConfig};

let config = DdimConfig {
    num_inference_steps: 30,
    eta: 0.0,  // 0.0 = deterministic, >0 = stochastic
};
let sampler = DdimSampler::new(schedule, config);
```

#### Euler

Simple first-order solver. Fast and stable.

```rust
use burn_image_samplers::{EulerSampler, EulerConfig};

let config = EulerConfig {
    num_inference_steps: 25,
    use_karras_sigmas: true,
    ..Default::default()
};
let sampler = EulerSampler::new(config, &schedule);
```

### Stochastic Samplers

These add randomness during sampling, producing more variety.

#### DPM++ SDE

DPM++ with stochastic differential equations. More variety than 2M.

```rust
use burn_image_samplers::{DpmPlusPlusSdeSampler, DpmConfig};

let sampler = DpmPlusPlusSdeSampler::new(DpmConfig {
    num_inference_steps: 25,
    use_karras_sigmas: true,
    ..Default::default()
}, &schedule);
```

#### Euler Ancestral

Euler with ancestral sampling. Good creative variety.

```rust
use burn_image_samplers::{EulerAncestralSampler, EulerConfig};

let sampler = EulerAncestralSampler::new(EulerConfig {
    num_inference_steps: 25,
    eta: 1.0,  // Controls randomness
    ..Default::default()
}, &schedule);
```

### Low-Step Samplers

Optimized for quality at very few steps.

#### LCM (Latent Consistency Model)

Extremely fast - works with 4-8 steps when using LCM-LoRA.

```rust
use burn_image_samplers::{LcmSampler, LcmConfig};

let config = LcmConfig {
    num_inference_steps: 4,
    guidance_scale: 1.0,  // LCM uses lower guidance
    ..Default::default()
};
let sampler = LcmSampler::new(config, &schedule);
```

::: warning
LCM requires LCM-LoRA weights or a model fine-tuned for LCM.
:::

#### UniPC

Unified predictor-corrector. Great quality at 10-15 steps.

```rust
use burn_image_samplers::{UniPcSampler, UniPcConfig};

let sampler = UniPcSampler::new(UniPcConfig {
    num_inference_steps: 15,
    order: 3,
    ..Default::default()
}, &schedule);
```

#### DEIS

Diffusion Exponential Integrator Sampler. Similar to UniPC.

```rust
use burn_image_samplers::{DeisSampler, DeisConfig, DeisSolverType};

let sampler = DeisSampler::new(DeisConfig {
    num_inference_steps: 15,
    order: 3,
    solver_type: DeisSolverType::LogRho,
    ..Default::default()
}, &schedule);
```

### Advanced Samplers

#### SA-Solver

Stochastic Adams solver with predictor-corrector.

```rust
use burn_image_samplers::{SaSolver, SaSolverConfig};

let sampler = SaSolver::new(SaSolverConfig {
    num_inference_steps: 25,
    predictor_order: 3,
    corrector_order: 2,
    use_karras_sigmas: true,
    ..Default::default()
}, &schedule);
```

#### Heun

Second-order Heun's method. Slower but higher quality.

```rust
use burn_image_samplers::{HeunSampler, HeunConfig};

let sampler = HeunSampler::new(HeunConfig {
    num_inference_steps: 25,
    ..Default::default()
}, &schedule);
```

## Karras Sigmas

Most samplers support Karras sigma scheduling, which improves quality:

```rust
let config = DpmConfig {
    use_karras_sigmas: true,  // Enable Karras schedule
    ..Default::default()
};
```

Karras sigmas concentrate more steps in the high-noise region, where they matter most.

## CFG++ (Classifier-Free Guidance Plus Plus)

CFG++ applies guidance in the denoised prediction space rather than noise space, reducing artifacts at high guidance scales.

```rust
use burn_image_samplers::{EulerCfgPlusPlusSampler, EulerCfgPlusPlusConfig};

let sampler = EulerCfgPlusPlusSampler::new(EulerCfgPlusPlusConfig {
    num_inference_steps: 25,
    guidance_rescale: 0.7,  // Prevents over-saturation
    ..Default::default()
}, &schedule);
```

Available for:
- `EulerCfgPlusPlusSampler`
- `EulerAncestralCfgPlusPlusSampler`
- `Dpm2mCfgPlusPlusSampler`
- `Dpm2sAncestralCfgPlusPlusSampler`

## Sigma Schedules

Sigma schedules control how noise levels are spaced during sampling.

| Schedule | Description | Best For |
|----------|-------------|----------|
| **Karras** | Concentrates steps at high noise (default) | General use |
| **Normal** | Uniform timestep spacing | Baseline/testing |
| **Exponential** | Exponential spacing in sigma | Smooth transitions |
| **SGM Uniform** | Uniform in sigma space | Score-based models |
| **Beta** | Beta distribution (more at high noise) | Quality priority |
| **Linear Quadratic** | Blended linear/quadratic | Balanced |

## Algorithm Details: DPM++ 2M Formulations

There are two main implementations of DPM++ 2M in the ecosystem:

### Original Paper (DPM-Solver++)

From Lu et al. "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models" (2022):

```
First-order:
  x_{t-1} = (σ_{t-1}/σ_t) * x_t + (1 - σ_{t-1}/σ_t) * D_0

Second-order:
  x_{t-1} = (σ_{t-1}/σ_t) * x_t + (1 - σ_{t-1}/σ_t) * D_0
          + (1 - σ_{t-1}/σ_t) * (h/2) * D_1

where:
  D_0 = denoised prediction (x_0 estimate)
  D_1 = (D_0 - D_0_prev) / r
  r = h / h_prev
  h = λ_{t-1} - λ_t  (step in log-SNR space)
  λ = -log(σ)
```

### k-diffusion (ComfyUI/A1111)

From crowsonkb's k-diffusion library:

```
First-order:
  x_{t-1} = (σ_{t-1}/σ_t) * x_t + (1 - exp(-h)) * D_0

Second-order:
  D_d = (1 + 1/(2r)) * D_0 - (1/(2r)) * D_0_prev
  x_{t-1} = (σ_{t-1}/σ_t) * x_t + (1 - exp(-h)) * D_d

where:
  r = h_prev / h  (INVERSE of paper convention)
```

### Key Differences

| Aspect | Paper | k-diffusion |
|--------|-------|-------------|
| Step coefficient | `(1 - σ_{t-1}/σ_t)` | `(1 - exp(-h))` |
| r definition | `h / h_prev` | `h_prev / h` |
| D_1 formula | `(D_0 - D_prev) / r` | `(1 + 1/(2r)) * D_0 - (1/(2r)) * D_prev` |

### Why We Use k-diffusion

This implementation uses the k-diffusion formulation because:

1. **ComfyUI/A1111 compatibility** - Users expect matching output with same seeds
2. **Numerical stability** - `exp(-h)` handles large Karras sigmas gracefully
3. **Battle-tested** - Millions of production generations

The formulations are mathematically equivalent, but k-diffusion's is more robust.

### References

- [DPM-Solver++ Paper](https://arxiv.org/abs/2211.01095)
- [k-diffusion](https://github.com/crowsonkb/k-diffusion)
- [diffusers](https://github.com/huggingface/diffusers)

## Recommendations

| Use Case | Recommended Sampler | Steps |
|----------|---------------------|-------|
| General use | DPM++ 2M Karras | 20 |
| Speed priority | LCM (with LCM-LoRA) | 4-6 |
| Quality priority | Heun or DPM++ SDE | 30+ |
| Low step count | UniPC or DEIS | 10-15 |
| Creative variety | Euler Ancestral | 25 |
| High CFG scale | CFG++ variants | 20-30 |
| Reproducibility | DDIM or Euler | 30 |

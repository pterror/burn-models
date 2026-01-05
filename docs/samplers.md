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

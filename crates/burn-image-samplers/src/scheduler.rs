//! Noise schedules for diffusion models
//!
//! This module provides noise schedule utilities shared across all samplers.

use burn::prelude::*;

// ============================================================================
// Schedule Configuration
// ============================================================================

/// Noise schedule configuration
#[derive(Debug, Clone)]
pub struct ScheduleConfig {
    /// Number of training timesteps
    pub num_train_steps: usize,
    /// Minimum signal rate for cosine schedule
    pub min_signal_rate: f64,
    /// Maximum signal rate for cosine schedule
    pub max_signal_rate: f64,
}

impl Default for ScheduleConfig {
    fn default() -> Self {
        Self {
            num_train_steps: 1000,
            min_signal_rate: 0.02,
            max_signal_rate: 0.95,
        }
    }
}

/// Precomputed noise schedule values
pub struct NoiseSchedule<B: Backend> {
    /// Cumulative product of alphas: ᾱₜ
    pub alphas_cumprod: Tensor<B, 1>,
    /// Number of training steps
    pub num_train_steps: usize,
}

impl<B: Backend> NoiseSchedule<B> {
    /// Create a linear beta schedule (used by SD 1.x)
    pub fn linear(num_steps: usize, beta_start: f64, beta_end: f64, device: &B::Device) -> Self {
        let betas: Vec<f32> = (0..num_steps)
            .map(|i| {
                let t = i as f64 / (num_steps - 1) as f64;
                (beta_start + t * (beta_end - beta_start)) as f32
            })
            .collect();

        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();

        // Cumulative product
        let mut alphas_cumprod = Vec::with_capacity(num_steps);
        let mut cumprod = 1.0f32;
        for alpha in alphas {
            cumprod *= alpha;
            alphas_cumprod.push(cumprod);
        }

        let data = TensorData::new(alphas_cumprod, [num_steps]);
        Self {
            alphas_cumprod: Tensor::from_data(data, device),
            num_train_steps: num_steps,
        }
    }

    /// Create an offset cosine schedule (used by SDXL)
    pub fn cosine(config: &ScheduleConfig, device: &B::Device) -> Self {
        let n = config.num_train_steps;
        let min_rate = config.min_signal_rate;
        let max_rate = config.max_signal_rate;

        let alphas_cumprod: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                let signal_rate = (1.0 - t) * max_rate + t * min_rate;
                let angle = signal_rate * std::f64::consts::FRAC_PI_2;
                (angle.cos().powi(2)) as f32
            })
            .collect();

        let data = TensorData::new(alphas_cumprod, [n]);
        Self {
            alphas_cumprod: Tensor::from_data(data, device),
            num_train_steps: n,
        }
    }

    /// Create the default SD 1.x schedule
    pub fn sd1x(device: &B::Device) -> Self {
        Self::linear(1000, 0.00085, 0.012, device)
    }

    /// Create the default SDXL schedule
    pub fn sdxl(device: &B::Device) -> Self {
        Self::cosine(&ScheduleConfig::default(), device)
    }

    /// Get alpha_cumprod at a specific timestep
    pub fn alpha_cumprod_at(&self, t: usize) -> Tensor<B, 1> {
        self.alphas_cumprod.clone().slice([t..t + 1])
    }

    /// Get sqrt(alpha_cumprod) at timestep
    pub fn sqrt_alpha_cumprod_at(&self, t: usize) -> Tensor<B, 1> {
        self.alpha_cumprod_at(t).sqrt()
    }

    /// Get sqrt(1 - alpha_cumprod) at timestep
    pub fn sqrt_one_minus_alpha_cumprod_at(&self, t: usize) -> Tensor<B, 1> {
        (self.alpha_cumprod_at(t).neg() + 1.0).sqrt()
    }
}

/// Generate timestep sequence for inference
pub fn inference_timesteps(num_inference_steps: usize, num_train_steps: usize) -> Vec<usize> {
    let step_ratio = num_train_steps / num_inference_steps;
    (0..num_inference_steps)
        .rev()
        .map(|i| i * step_ratio)
        .collect()
}

// ============================================================================
// Sigma Utilities (shared across samplers)
// ============================================================================

/// Compute sigmas from timesteps using the noise schedule
///
/// Converts alpha_cumprod values to sigma = sqrt((1 - alpha) / alpha)
pub fn sigmas_from_timesteps<B: Backend>(schedule: &NoiseSchedule<B>, timesteps: &[usize]) -> Vec<f32> {
    timesteps
        .iter()
        .map(|&t| {
            let alpha_cumprod = schedule.alpha_cumprod_at(t);
            let alpha_data = alpha_cumprod.into_data();
            let alpha: f32 = alpha_data.to_vec().unwrap()[0];
            ((1.0 - alpha) / alpha).sqrt()
        })
        .collect()
}

/// Apply Karras noise schedule transformation
///
/// Transforms sigmas using the Karras et al. schedule for improved sampling.
/// Uses rho=7.0 as recommended in the paper.
pub fn apply_karras_schedule(sigmas: &[f32], rho: f32) -> Vec<f32> {
    if sigmas.is_empty() {
        return Vec::new();
    }

    let sigma_min = *sigmas.last().unwrap_or(&0.0);
    let sigma_max = *sigmas.first().unwrap_or(&1.0);
    let n = sigmas.len();

    (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1).max(1) as f32;
            let sigma_min_inv_rho = sigma_min.powf(1.0 / rho);
            let sigma_max_inv_rho = sigma_max.powf(1.0 / rho);
            (sigma_max_inv_rho + t * (sigma_min_inv_rho - sigma_max_inv_rho)).powf(rho)
        })
        .collect()
}

/// Compute sigmas with optional Karras schedule
///
/// This is the main entry point for computing sigmas in samplers.
/// Appends sigma=0.0 at the end for the final denoising step.
pub fn compute_sigmas<B: Backend>(
    schedule: &NoiseSchedule<B>,
    timesteps: &[usize],
    use_karras: bool,
) -> Vec<f32> {
    let mut sigmas = sigmas_from_timesteps(schedule, timesteps);

    if use_karras {
        sigmas = apply_karras_schedule(&sigmas, 7.0);
    }

    sigmas.push(0.0);
    sigmas
}

/// Compute ancestral sampling step parameters
///
/// For stochastic samplers (ancestral, SDE), computes:
/// - sigma_down: the deterministic step target
/// - sigma_up: the noise injection level
///
/// The eta parameter controls stochasticity (0 = ODE, 1 = full SDE)
pub fn get_ancestral_step(sigma: f32, sigma_next: f32, eta: f32) -> (f32, f32) {
    if sigma_next == 0.0 {
        return (0.0, 0.0);
    }

    let sigma_up = (sigma_next.powi(2) * (sigma.powi(2) - sigma_next.powi(2)) / sigma.powi(2))
        .sqrt()
        .min(sigma_next)
        * eta;
    let sigma_down = (sigma_next.powi(2) - sigma_up.powi(2)).sqrt();

    (sigma_down, sigma_up)
}

/// Generate timesteps for a sampler
///
/// Creates evenly spaced timesteps from high noise to low noise.
pub fn sampler_timesteps(num_inference_steps: usize, num_train_steps: usize) -> Vec<usize> {
    let step_ratio = num_train_steps / num_inference_steps;
    (0..num_inference_steps)
        .rev()
        .map(|i| (i * step_ratio).min(num_train_steps - 1))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_timesteps() {
        let steps = inference_timesteps(50, 1000);
        assert_eq!(steps.len(), 50);
        assert_eq!(steps[0], 980); // First step (highest noise)
        assert_eq!(steps[49], 0); // Last step (lowest noise)
    }
}

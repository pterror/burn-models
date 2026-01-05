//! DPM++ (Diffusion Probabilistic Model) Samplers
//!
//! Implements DPM-Solver++ for high-quality, fast sampling.
//! These samplers typically produce excellent results in 15-25 steps.

use burn::prelude::*;

use crate::scheduler::{NoiseSchedule, sampler_timesteps, sigmas_from_timesteps, init_noise_latent};

/// DPM++ configuration
#[derive(Debug, Clone)]
pub struct DpmConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Solver order (1 = Euler-like, 2 = second-order)
    pub solver_order: usize,
}

impl Default for DpmConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 25,
            solver_order: 2,
        }
    }
}

/// DPM++ 2M Sampler (second-order multistep)
///
/// High-quality sampler that produces excellent results in ~20 steps.
/// Uses a second-order multistep method for better accuracy.
pub struct DpmPlusPlusSampler<B: Backend> {
    schedule: NoiseSchedule<B>,
    config: DpmConfig,
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    /// Previous model output for multistep
    prev_sample: Option<Tensor<B, 4>>,
    /// Previous sigma
    prev_sigma: Option<f32>,
}

impl<B: Backend> DpmPlusPlusSampler<B> {
    /// Create a new DPM++ 2M sampler
    pub fn new(schedule: NoiseSchedule<B>, config: DpmConfig, _device: &B::Device) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let mut sigmas = sigmas_from_timesteps(&schedule, &timesteps);
        sigmas.push(0.0);

        Self {
            schedule,
            config,
            timesteps,
            sigmas,
            prev_sample: None,
            prev_sigma: None,
        }
    }

    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    pub fn num_steps(&self) -> usize {
        self.config.num_inference_steps
    }

    /// Reset state for new generation
    pub fn reset(&mut self) {
        self.prev_sample = None;
        self.prev_sigma = None;
    }

    /// Perform one DPM++ 2M step
    pub fn step(
        &mut self,
        latent: Tensor<B, 4>,
        noise_pred: Tensor<B, 4>,
        step_index: usize,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[step_index];
        let sigma_next = self.sigmas[step_index + 1];

        // Convert to log domain for numerical stability
        let lambda_t = -(sigma.ln());
        let lambda_next = if sigma_next > 0.0 {
            -(sigma_next.ln())
        } else {
            f32::INFINITY
        };

        let h = lambda_next - lambda_t;

        // Compute denoised estimate (x0 prediction)
        let denoised = latent.clone() - noise_pred.clone() * sigma;

        if self.config.solver_order == 1 || self.prev_sample.is_none() {
            // First-order (Euler-like)
            let result = if sigma_next > 0.0 {
                // x_{t-1} = sigma_{t-1} / sigma_t * x_t + (1 - sigma_{t-1}/sigma_t) * denoised
                let ratio = sigma_next / sigma;
                latent.clone() * ratio + denoised.clone() * (1.0 - ratio)
            } else {
                denoised.clone()
            };

            self.prev_sample = Some(denoised);
            self.prev_sigma = Some(sigma);

            result
        } else {
            // Second-order (multistep)
            let prev_denoised = self.prev_sample.take().unwrap();
            let prev_sigma = self.prev_sigma.take().unwrap();

            let lambda_prev = -(prev_sigma.ln());
            let h_prev = lambda_t - lambda_prev;
            let r = h / h_prev;

            // Second-order correction
            let d0 = denoised.clone();
            let d1 = (denoised.clone() - prev_denoised) / r;

            let result = if sigma_next > 0.0 {
                let ratio = sigma_next / sigma;
                latent.clone() * ratio + d0.clone() * (1.0 - ratio) + d1 * (1.0 - ratio) * (h / 2.0)
            } else {
                d0.clone() + d1 * (h / 2.0)
            };

            self.prev_sample = Some(d0);
            self.prev_sigma = Some(sigma);

            result
        }
    }

    pub fn init_latent(
        &self,
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        init_noise_latent(batch_size, channels, height, width, self.sigmas[0], device)
    }
}

/// DPM++ SDE Sampler (stochastic differential equation variant)
///
/// Adds controlled noise during sampling for more diverse results.
/// Good for creative generation with ~25-30 steps.
pub struct DpmPlusPlusSdeSampler<B: Backend> {
    schedule: NoiseSchedule<B>,
    config: DpmConfig,
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    /// Noise multiplier (0.0 = deterministic, 1.0 = full noise)
    eta: f32,
}

impl<B: Backend> DpmPlusPlusSdeSampler<B> {
    /// Create a new DPM++ SDE sampler
    pub fn new(schedule: NoiseSchedule<B>, config: DpmConfig, eta: f32, _device: &B::Device) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let mut sigmas = sigmas_from_timesteps(&schedule, &timesteps);
        sigmas.push(0.0);

        Self {
            schedule,
            config,
            timesteps,
            sigmas,
            eta,
        }
    }

    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    pub fn num_steps(&self) -> usize {
        self.config.num_inference_steps
    }

    /// Perform one DPM++ SDE step
    pub fn step(
        &self,
        latent: Tensor<B, 4>,
        noise_pred: Tensor<B, 4>,
        step_index: usize,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[step_index];
        let sigma_next = self.sigmas[step_index + 1];

        if sigma_next == 0.0 {
            // Last step: just denoise
            return latent.clone() - noise_pred * sigma;
        }

        // Compute noise to inject
        let sigma_up = (self.eta * sigma_next.powi(2) * (sigma.powi(2) - sigma_next.powi(2)) / sigma.powi(2)).sqrt();
        let sigma_down = (sigma_next.powi(2) - sigma_up.powi(2)).sqrt();

        // Compute denoised
        let denoised = latent.clone() - noise_pred * sigma;

        // DPM step to sigma_down
        let ratio = sigma_down / sigma;
        let latent_down = latent * ratio + denoised * (1.0 - ratio);

        // Add noise
        let noise: Tensor<B, 4> = Tensor::random(
            latent_down.shape(),
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &latent_down.device(),
        );

        latent_down + noise * sigma_up
    }

    pub fn init_latent(
        &self,
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        init_noise_latent(batch_size, channels, height, width, self.sigmas[0], device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpm_config_default() {
        let config = DpmConfig::default();
        assert_eq!(config.num_inference_steps, 25);
        assert_eq!(config.solver_order, 2);
    }
}

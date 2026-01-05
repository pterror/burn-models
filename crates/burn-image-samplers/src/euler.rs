//! Euler sampler for diffusion models
//!
//! Implements the simple Euler method for ODE-based sampling.
//! Fast and produces good results with ~20-30 steps.

use burn::prelude::*;

use crate::scheduler::{NoiseSchedule, sampler_timesteps, sigmas_from_timesteps, init_noise_latent};

/// Euler sampler configuration
#[derive(Debug, Clone)]
pub struct EulerConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
}

impl Default for EulerConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 30,
        }
    }
}

/// Euler Sampler (ancestral variant)
///
/// Uses the Euler method to solve the diffusion ODE.
/// This is faster than DDIM and often produces good results.
pub struct EulerSampler<B: Backend> {
    schedule: NoiseSchedule<B>,
    config: EulerConfig,
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
}

impl<B: Backend> EulerSampler<B> {
    /// Create a new Euler sampler
    pub fn new(schedule: NoiseSchedule<B>, config: EulerConfig, _device: &B::Device) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let mut sigmas = sigmas_from_timesteps(&schedule, &timesteps);
        sigmas.push(0.0);

        Self {
            schedule,
            config,
            timesteps,
            sigmas,
        }
    }

    /// Get the timesteps for inference
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Get the number of inference steps
    pub fn num_steps(&self) -> usize {
        self.config.num_inference_steps
    }

    /// Perform one Euler step
    pub fn step(
        &self,
        latent: Tensor<B, 4>,
        noise_pred: Tensor<B, 4>,
        step_index: usize,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[step_index];
        let sigma_next = self.sigmas[step_index + 1];

        // Compute dt
        let dt = sigma_next - sigma;

        // Euler step: x_next = x + dt * dx/dt
        // For diffusion: dx/dt = (x - denoised) / sigma
        // Where denoised = x - sigma * noise_pred

        // Compute denoised estimate
        let denoised = latent.clone() - noise_pred.clone() * sigma;

        // Derivative
        let derivative = (latent.clone() - denoised) / sigma;

        // Euler step
        latent + derivative * dt
    }

    /// Initialize latent with random noise scaled appropriately
    pub fn init_latent(
        &self,
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let sigma_max = self.sigmas[0];
        let noise: Tensor<B, 4> = Tensor::random(
            [batch_size, channels, height, width],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        noise * sigma_max
    }
}

/// Euler Ancestral sampler
///
/// Adds noise during sampling for more stochastic results.
/// Often produces more creative outputs.
pub struct EulerAncestralSampler<B: Backend> {
    schedule: NoiseSchedule<B>,
    config: EulerConfig,
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
}

impl<B: Backend> EulerAncestralSampler<B> {
    /// Create a new Euler Ancestral sampler
    pub fn new(schedule: NoiseSchedule<B>, config: EulerConfig, _device: &B::Device) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let mut sigmas = sigmas_from_timesteps(&schedule, &timesteps);
        sigmas.push(0.0);

        Self {
            schedule,
            config,
            timesteps,
            sigmas,
        }
    }

    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    pub fn num_steps(&self) -> usize {
        self.config.num_inference_steps
    }

    /// Perform one Euler Ancestral step (with noise injection)
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

        // Compute sigma_up and sigma_down
        let sigma_up = (sigma_next.powi(2) * (sigma.powi(2) - sigma_next.powi(2)) / sigma.powi(2)).sqrt();
        let sigma_down = (sigma_next.powi(2) - sigma_up.powi(2)).sqrt();

        // Compute denoised
        let denoised = latent.clone() - noise_pred.clone() * sigma;

        // Euler step to sigma_down
        let dt = sigma_down - sigma;
        let derivative = (latent.clone() - denoised.clone()) / sigma;
        let latent_down = latent + derivative * dt;

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
    fn test_euler_config_default() {
        let config = EulerConfig::default();
        assert_eq!(config.num_inference_steps, 30);
    }
}

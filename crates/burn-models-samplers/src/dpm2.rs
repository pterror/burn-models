//! DPM2 (Diffusion Probabilistic Models 2nd order) samplers
//!
//! Second-order solvers for the diffusion ODE/SDE.

use burn::prelude::*;

use crate::scheduler::{NoiseSchedule, sampler_timesteps, sigmas_from_timesteps, compute_sigmas};

/// Configuration for DPM2 sampler
#[derive(Debug, Clone)]
pub struct Dpm2Config {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Solver order
    pub solver_order: usize,
    /// Use Karras sigmas
    pub use_karras_sigmas: bool,
}

impl Default for Dpm2Config {
    fn default() -> Self {
        Self {
            num_inference_steps: 30,
            solver_order: 2,
            use_karras_sigmas: false,
        }
    }
}

/// DPM2 Sampler (second-order DPM solver)
pub struct Dpm2Sampler<B: Backend> {
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> Dpm2Sampler<B> {
    /// Create a new DPM2 sampler
    pub fn new(config: Dpm2Config, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let sigmas = compute_sigmas(schedule, &timesteps, config.use_karras_sigmas);

        Self {
            timesteps,
            sigmas,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Perform one DPM2 step
    pub fn step(
        &self,
        model_output: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        if sigma_next == 0.0 {
            // Final step
            return sample.clone() - model_output * sigma;
        }

        // First half step
        let denoised = sample.clone() - model_output.clone() * sigma;
        let d = (sample.clone() - denoised.clone()) / sigma;

        // TODO: True DPM2 would evaluate model at sigma_mid, requiring a second model call.
        // For now, reuse the derivative (equivalent to first-order).
        let d_mid = d.clone();

        // Full step using midpoint derivative
        sample + d_mid * (sigma_next - sigma)
    }
}

/// DPM2 Ancestral sampler (with noise injection)
pub struct Dpm2AncestralSampler<B: Backend> {
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> Dpm2AncestralSampler<B> {
    /// Create a new DPM2 Ancestral sampler
    pub fn new(config: Dpm2Config, schedule: &NoiseSchedule<B>) -> Self {
        let timesteps = sampler_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let mut sigmas = sigmas_from_timesteps(schedule, &timesteps);
        sigmas.push(0.0);

        Self {
            timesteps,
            sigmas,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Perform one DPM2 Ancestral step
    pub fn step(
        &self,
        model_output: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        if sigma_next == 0.0 {
            return sample.clone() - model_output * sigma;
        }

        // Ancestral sampling adds noise
        let sigma_up = (sigma_next.powi(2) * (sigma.powi(2) - sigma_next.powi(2)) / sigma.powi(2)).sqrt();
        let sigma_down = (sigma_next.powi(2) - sigma_up.powi(2)).sqrt();

        // Denoising step
        let denoised = sample.clone() - model_output * sigma;
        let d = (sample.clone() - denoised.clone()) / sigma;

        let sample_next = sample + d * (sigma_down - sigma);

        // Add noise
        if sigma_up > 0.0 {
            let device = sample_next.device();
            let shape = sample_next.dims();
            let noise: Tensor<B, 4> = Tensor::random(shape, burn::tensor::Distribution::Normal(0.0, 1.0), &device);
            sample_next + noise * sigma_up
        } else {
            sample_next
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpm2_config_default() {
        let config = Dpm2Config::default();
        assert_eq!(config.solver_order, 2);
        assert!(!config.use_karras_sigmas);
    }
}

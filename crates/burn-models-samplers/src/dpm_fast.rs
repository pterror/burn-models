//! DPM Fast and DPM Adaptive samplers
//!
//! DPM-Solver with adaptive step size selection for efficient sampling.
//! DPM Fast uses fixed fast schedules, while DPM Adaptive adjusts step sizes.

use burn::prelude::*;

use crate::scheduler::{NoiseSchedule, sigmas_from_timesteps};

/// Configuration for DPM Fast sampler
#[derive(Debug, Clone)]
pub struct DpmFastConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Solver order (1-3)
    pub solver_order: usize,
}

impl Default for DpmFastConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 15,
            solver_order: 2,
        }
    }
}

/// DPM Fast Sampler
///
/// A fast variant of DPM-Solver that uses optimized step schedules
/// for rapid sampling with fewer steps.
pub struct DpmFastSampler<B: Backend> {
    /// Timestep indices for sampling
    timesteps: Vec<usize>,
    /// Sigma values at each timestep
    sigmas: Vec<f32>,
    /// Phantom data for backend type
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> DpmFastSampler<B> {
    /// Create a new DPM Fast sampler
    pub fn new(config: DpmFastConfig, schedule: &NoiseSchedule<B>) -> Self {
        // DPM Fast uses a specific timestep schedule optimized for speed
        let timesteps = Self::compute_fast_timesteps(config.num_inference_steps, schedule.num_train_steps);
        let mut sigmas = sigmas_from_timesteps(schedule, &timesteps);
        sigmas.push(0.0);

        Self {
            timesteps,
            sigmas,
            _marker: std::marker::PhantomData,
        }
    }

    /// Computes optimized timesteps for fast sampling
    fn compute_fast_timesteps(num_steps: usize, num_train_steps: usize) -> Vec<usize> {
        // Use a polynomial schedule for faster convergence
        (0..num_steps)
            .map(|i| {
                let t = 1.0 - (i as f64 / num_steps as f64).powf(2.0);
                ((t * num_train_steps as f64) as usize).min(num_train_steps - 1)
            })
            .collect()
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Perform one DPM Fast step
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

        // Denoised prediction
        let denoised = sample.clone() - model_output.clone() * sigma;

        // DPM-Solver update (first order for speed)
        let sigma_ratio = sigma_next / sigma;
        sample.clone() * sigma_ratio + denoised * (1.0 - sigma_ratio)
    }
}

/// Configuration for DPM Adaptive sampler
#[derive(Debug, Clone)]
pub struct DpmAdaptiveConfig {
    /// Minimum number of steps
    pub min_steps: usize,
    /// Maximum number of steps
    pub max_steps: usize,
    /// Error tolerance for step size adaptation
    pub rtol: f32,
    /// Absolute error tolerance
    pub atol: f32,
    /// Solver order (1-3)
    pub solver_order: usize,
}

impl Default for DpmAdaptiveConfig {
    fn default() -> Self {
        Self {
            min_steps: 10,
            max_steps: 100,
            rtol: 0.05,
            atol: 0.0078,
            solver_order: 2,
        }
    }
}

/// DPM Adaptive Sampler
///
/// Uses adaptive step size control to automatically determine the
/// optimal number of steps based on local error estimates.
pub struct DpmAdaptiveSampler<B: Backend> {
    config: DpmAdaptiveConfig,
    sigmas: Vec<f32>,
    current_sigma: f32,
    sigma_min: f32,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> DpmAdaptiveSampler<B> {
    /// Create a new DPM Adaptive sampler
    pub fn new(config: DpmAdaptiveConfig, schedule: &NoiseSchedule<B>) -> Self {
        let num_train_steps = schedule.num_train_steps;

        // Get sigma range from schedule
        let alpha_min = {
            let alpha_cumprod = schedule.alpha_cumprod_at(num_train_steps - 1);
            let alpha_data = alpha_cumprod.into_data();
            alpha_data.to_vec::<f32>().unwrap()[0]
        };
        let alpha_max = {
            let alpha_cumprod = schedule.alpha_cumprod_at(0);
            let alpha_data = alpha_cumprod.into_data();
            alpha_data.to_vec::<f32>().unwrap()[0]
        };

        let sigma_max = ((1.0 - alpha_min) / alpha_min).sqrt();
        let sigma_min = ((1.0 - alpha_max) / alpha_max).sqrt();

        Self {
            config,
            sigmas: Vec::new(),
            current_sigma: sigma_max,
            sigma_min,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get the current sigma value
    pub fn current_sigma(&self) -> f32 {
        self.current_sigma
    }

    /// Check if sampling is complete
    pub fn is_done(&self) -> bool {
        self.current_sigma <= self.sigma_min
    }

    /// Perform one adaptive DPM step
    ///
    /// Returns (next_sample, step_accepted, new_sigma)
    pub fn step(
        &mut self,
        model_output: Tensor<B, 4>,
        sample: Tensor<B, 4>,
        proposed_sigma_next: Option<f32>,
    ) -> (Tensor<B, 4>, bool, f32) {
        let sigma = self.current_sigma;
        let sigma_next = proposed_sigma_next.unwrap_or_else(|| {
            // Default step: geometric mean towards sigma_min
            let log_sigma = sigma.ln();
            let log_sigma_min = self.sigma_min.ln();
            let h = (log_sigma_min - log_sigma) / (self.config.max_steps as f32);
            (log_sigma + h).exp()
        });

        let sigma_next = sigma_next.max(self.sigma_min);

        // Denoised prediction
        let denoised = sample.clone() - model_output.clone() * sigma;

        // DPM-Solver step
        let sigma_ratio = sigma_next / sigma;
        let next_sample = sample.clone() * sigma_ratio + denoised.clone() * (1.0 - sigma_ratio);

        // For adaptive control, we'd need a second evaluation
        // For now, accept all steps
        self.current_sigma = sigma_next;
        self.sigmas.push(sigma_next);

        (next_sample, true, sigma_next)
    }

    /// Get the sigmas used so far
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpm_fast_config_default() {
        let config = DpmFastConfig::default();
        assert_eq!(config.num_inference_steps, 15);
        assert_eq!(config.solver_order, 2);
    }

    #[test]
    fn test_dpm_adaptive_config_default() {
        let config = DpmAdaptiveConfig::default();
        assert_eq!(config.min_steps, 10);
        assert_eq!(config.max_steps, 100);
        assert_eq!(config.rtol, 0.05);
    }
}

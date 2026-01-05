//! DEIS (Diffusion Exponential Integrator Sampler)
//!
//! A fast sampler based on exponential integrators for diffusion ODEs.
//! Uses polynomial extrapolation for improved accuracy.

use burn::prelude::*;
use std::collections::VecDeque;

use crate::scheduler::NoiseSchedule;

/// Configuration for DEIS sampler
#[derive(Debug, Clone)]
pub struct DeisConfig {
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// Order of the method (1-4)
    pub order: usize,
    /// Solver type
    pub solver_type: DeisSolverType,
}

/// DEIS solver type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeisSolverType {
    /// Logarithmic rho schedule (default, better for most cases)
    #[default]
    LogRho,
    /// Polynomial interpolation
    Polynomial,
}

impl Default for DeisConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 20,
            order: 3,
            solver_type: DeisSolverType::LogRho,
        }
    }
}

/// DEIS sampler
///
/// Diffusion Exponential Integrator Sampler uses exponential integrators
/// combined with polynomial extrapolation for fast, high-quality sampling.
pub struct DeisSampler<B: Backend> {
    config: DeisConfig,
    timesteps: Vec<usize>,
    sigmas: Vec<f32>,
    log_sigmas: Vec<f32>,
    /// History of model outputs for multi-step methods
    model_outputs: VecDeque<Tensor<B, 4>>,
}

impl<B: Backend> DeisSampler<B> {
    /// Create a new DEIS sampler
    pub fn new(config: DeisConfig, schedule: &NoiseSchedule<B>) -> Self {
        let num_train_steps = schedule.num_train_steps;
        let step_ratio = num_train_steps / config.num_inference_steps;

        let timesteps: Vec<usize> = (0..config.num_inference_steps)
            .rev()
            .map(|i| (i * step_ratio).min(num_train_steps - 1))
            .collect();

        let sigmas = Self::compute_sigmas(schedule, &timesteps);
        let log_sigmas: Vec<f32> = sigmas.iter().map(|s| (s + 1e-10).ln()).collect();

        Self {
            config,
            timesteps,
            sigmas,
            log_sigmas,
            model_outputs: VecDeque::new(),
        }
    }

    fn compute_sigmas(schedule: &NoiseSchedule<B>, timesteps: &[usize]) -> Vec<f32> {
        let mut sigmas: Vec<f32> = timesteps
            .iter()
            .map(|&t| {
                let alpha_cumprod = schedule.alpha_cumprod_at(t);
                let alpha_data = alpha_cumprod.into_data();
                let alpha: f32 = alpha_data.to_vec().unwrap()[0];
                ((1.0 - alpha) / alpha).sqrt()
            })
            .collect();
        sigmas.push(0.0);
        sigmas
    }

    /// Get the timesteps
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Reset the model output history
    pub fn reset(&mut self) {
        self.model_outputs.clear();
    }

    /// Compute DEIS coefficients for polynomial interpolation
    fn get_deis_coefficients(&self, order: usize, step_idx: usize) -> Vec<f32> {
        let mut coeffs = vec![0.0; order];
        let h = self.log_sigmas[step_idx + 1] - self.log_sigmas[step_idx];

        match order {
            1 => {
                // First order (Euler)
                coeffs[0] = 1.0;
            }
            2 => {
                // Second order
                if step_idx >= 1 {
                    let h_0 = self.log_sigmas[step_idx] - self.log_sigmas[step_idx - 1];
                    let r = h / h_0;
                    coeffs[0] = 1.0 + 0.5 * r;
                    coeffs[1] = -0.5 * r;
                } else {
                    coeffs[0] = 1.0;
                }
            }
            3 => {
                // Third order
                if step_idx >= 2 {
                    let h_0 = self.log_sigmas[step_idx] - self.log_sigmas[step_idx - 1];
                    let h_1 = self.log_sigmas[step_idx - 1] - self.log_sigmas[step_idx - 2];
                    let r_0 = h / h_0;
                    let r_1 = h_0 / h_1;

                    coeffs[0] = 1.0 + r_0 * (1.0 + r_0) / (1.0 + r_1) / 2.0;
                    coeffs[1] = -r_0 * (1.0 + r_0 + r_1) / (1.0 + r_1) / 2.0;
                    coeffs[2] = r_0 * r_0 * r_1 / (1.0 + r_1) / 2.0;
                } else if step_idx >= 1 {
                    let h_0 = self.log_sigmas[step_idx] - self.log_sigmas[step_idx - 1];
                    let r = h / h_0;
                    coeffs[0] = 1.0 + 0.5 * r;
                    coeffs[1] = -0.5 * r;
                } else {
                    coeffs[0] = 1.0;
                }
            }
            _ => {
                // Fourth order (simplified)
                if step_idx >= 3 {
                    // Use third order coefficients with correction
                    coeffs = self.get_deis_coefficients(3, step_idx);
                    coeffs.resize(4, 0.0);
                } else {
                    coeffs = self.get_deis_coefficients(step_idx.min(3) + 1, step_idx);
                }
            }
        }

        coeffs
    }

    /// Perform one DEIS step
    pub fn step(
        &mut self,
        model_output: Tensor<B, 4>,
        timestep_idx: usize,
        sample: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];

        // Store model output
        self.model_outputs.push_front(model_output.clone());
        if self.model_outputs.len() > self.config.order {
            self.model_outputs.pop_back();
        }

        // Compute denoised prediction
        let denoised = sample.clone() - model_output.clone() * sigma;

        if sigma_next == 0.0 {
            return denoised;
        }

        // Determine effective order
        let order = self.model_outputs.len().min(self.config.order);
        let coeffs = self.get_deis_coefficients(order, timestep_idx);

        // Compute weighted combination of derivatives
        let h = (sigma_next / sigma).ln();

        let mut derivative = Tensor::zeros(sample.dims(), &sample.device());
        for (i, coeff) in coeffs.iter().enumerate().take(order) {
            if let Some(output) = self.model_outputs.get(i) {
                derivative = derivative + output.clone() * *coeff;
            }
        }

        // Apply the step
        let sigma_ratio = sigma_next / sigma;
        sample.clone() * sigma_ratio + derivative * (sigma_next - sigma * sigma_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deis_config_default() {
        let config = DeisConfig::default();
        assert_eq!(config.order, 3);
        assert_eq!(config.num_inference_steps, 20);
    }
}

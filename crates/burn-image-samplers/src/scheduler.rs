//! Noise schedules for diffusion models

use burn::prelude::*;

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

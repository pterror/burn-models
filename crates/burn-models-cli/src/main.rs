//! burn-models CLI
//!
//! Command-line interface for model inference in pure Rust.
//!
//! Supports:
//! - Stable Diffusion image generation
//! - LLM text generation

use anyhow::{Context, Result};
use burn::prelude::*;
use clap::{Parser, Subcommand, ValueEnum};
use image::{ImageBuffer, Rgb};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

/// Float precision for inference
///
/// Compile with specific precision features to reduce binary size:
/// - `precision-f32`: f32 support (slowest, most stable)
/// - `precision-f16`: f16 support (fast, may NaN without flash attention)
/// - `precision-bf16`: bf16 support (fast, stable on Ampere+)
/// - `precision-all`: all precisions (default)
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Precision {
    /// 32-bit float (more VRAM, most stable, no JIT overhead)
    #[cfg(feature = "precision-f32")]
    F32,
    /// 16-bit float (less VRAM, may produce NaN without flash attention)
    #[cfg(feature = "precision-f16")]
    F16,
    /// Brain float16 (same range as f32, fast on Ampere+ GPUs)
    #[cfg(feature = "precision-bf16")]
    Bf16,
}

impl Default for Precision {
    fn default() -> Self {
        // Prefer bf16 > f16 > f32 as default (best balance of speed/stability)
        #[cfg(feature = "precision-bf16")]
        {
            return Precision::Bf16;
        }
        #[cfg(all(feature = "precision-f16", not(feature = "precision-bf16")))]
        {
            return Precision::F16;
        }
        #[cfg(all(
            feature = "precision-f32",
            not(feature = "precision-f16"),
            not(feature = "precision-bf16")
        ))]
        {
            return Precision::F32;
        }
        #[allow(unreachable_code)]
        {
            panic!(
                "No precision feature enabled. Enable precision-f32, precision-f16, or precision-bf16."
            )
        }
    }
}

/// Compute device for inference
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum Device {
    /// Auto-detect best available (CUDA > WGPU > CPU)
    #[default]
    Auto,
    /// NVIDIA CUDA GPU
    #[cfg(feature = "cuda")]
    Cuda,
    /// WebGPU (Vulkan/Metal/DX12)
    #[cfg(feature = "wgpu")]
    Wgpu,
    /// CPU (CubeCL CPU backend)
    #[cfg(feature = "cpu")]
    Cpu,
}

use burn_models_clip::{ClipConfig, ClipTokenizer};
use burn_models_convert::sd_loader::SdWeightLoader;
use burn_models_samplers::NoiseSchedule;
use burn_models_unet::UNetConfig;
use burn_models_vae::DecoderConfig;

mod llm;

#[derive(Parser)]
#[command(name = "burn-models")]
#[command(about = "Model inference in pure Rust (Stable Diffusion, LLMs)")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// LLM text generation commands
    Llm {
        #[command(subcommand)]
        command: llm::LlmCommands,
    },

    /// Generate an image from a text prompt (Stable Diffusion)
    Generate {
        /// Text prompt describing the desired image
        #[arg(short, long)]
        prompt: String,

        /// Negative prompt (things to avoid)
        #[arg(short, long, default_value = "")]
        negative: String,

        /// Output image path
        #[arg(short, long, default_value = "output.png")]
        output: PathBuf,

        /// Model type to use
        #[arg(short, long, value_enum, default_value = "sdxl")]
        model: ModelType,

        /// Image width (default: model native - 512 for SD1x, 1024 for SDXL)
        #[arg(long)]
        width: Option<usize>,

        /// Image height (default: model native - 512 for SD1x, 1024 for SDXL)
        #[arg(long)]
        height: Option<usize>,

        /// Number of inference steps
        #[arg(long, default_value = "30")]
        steps: usize,

        /// Guidance scale
        #[arg(long, default_value = "7.5")]
        guidance: f64,

        /// Random seed (optional)
        #[arg(long)]
        seed: Option<u64>,

        /// Path to vocabulary file (uses embedded CLIP vocab if not specified)
        #[arg(long)]
        vocab: Option<PathBuf>,

        /// Path to model weights directory or safetensors file
        #[arg(long)]
        weights: PathBuf,

        /// LoRA model paths (can be specified multiple times)
        #[arg(long = "lora", value_name = "FILE")]
        loras: Vec<PathBuf>,

        /// LoRA scales (same order as --lora, default 1.0)
        #[arg(long = "lora-scale", value_name = "SCALE")]
        lora_scales: Vec<f64>,

        /// Float precision (f32 = more VRAM, f16 = less VRAM, faster after warmup)
        #[arg(long, value_enum, default_value = "f16")]
        precision: Precision,

        /// Compute device (auto = detect best available)
        #[arg(long, value_enum, default_value = "auto")]
        device: Device,

        /// Debug modes (comma-separated): timing, shapes, nan, sampler, all
        /// - timing: Show timing for each step
        /// - shapes: Show model weight shapes
        /// - nan: Panic on NaN/Inf values (helps find f16 overflow)
        /// - sampler: Show sampler steps, sigmas, and latent stats
        ///
        /// Examples: --debug  --debug timing  --debug sampler,nan
        #[arg(long, value_delimiter = ',', num_args = 0.., default_missing_value = "all")]
        debug: Vec<String>,

        /// Enable flash attention (required for f16 without NaN overflow)
        /// Flash attention uses f32 accumulation internally, preventing overflow
        /// while still storing weights in f16 for memory savings.
        /// Enabled by default. Use --no-flash-attention to disable.
        /// Requires: CUDA or WGPU backend with cubecl feature enabled.
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        flash_attention: bool,

        /// Sampler/scheduler algorithm
        #[arg(long, value_enum, default_value = "dpm-plus-plus")]
        sampler: SamplerType,

        /// Noise schedule type
        #[arg(long, value_enum, default_value = "karras")]
        schedule: ScheduleType,

        /// Denoising strength (1.0 = full generation, <1.0 = partial denoising)
        /// Useful for img2img-like workflows with initial noise
        #[arg(long, default_value = "1.0")]
        denoise: f64,
    },

    /// Transform an existing image based on a prompt
    Img2Img {
        /// Input image path
        #[arg(short, long)]
        input: PathBuf,

        /// Text prompt for transformation
        #[arg(short, long)]
        prompt: String,

        /// Negative prompt
        #[arg(short, long, default_value = "")]
        negative: String,

        /// Output image path
        #[arg(short, long, default_value = "output.png")]
        output: PathBuf,

        /// Model type to use
        #[arg(short, long, value_enum, default_value = "sdxl")]
        model: ModelType,

        /// Transformation strength (0.0 = no change, 1.0 = full regeneration)
        #[arg(long, default_value = "0.75")]
        strength: f64,

        /// Number of inference steps
        #[arg(long, default_value = "30")]
        steps: usize,

        /// Guidance scale
        #[arg(long, default_value = "7.5")]
        guidance: f64,

        /// Path to vocabulary file (uses embedded CLIP vocab if not specified)
        #[arg(long)]
        vocab: Option<PathBuf>,

        /// Path to model weights directory or safetensors file
        #[arg(long)]
        weights: PathBuf,
    },

    /// Inpaint masked regions of an image
    Inpaint {
        /// Input image path
        #[arg(short, long)]
        input: PathBuf,

        /// Mask image path (white = regenerate, black = preserve)
        #[arg(short, long)]
        mask: PathBuf,

        /// Text prompt for inpainted regions
        #[arg(short, long)]
        prompt: String,

        /// Negative prompt
        #[arg(short, long, default_value = "")]
        negative: String,

        /// Output image path
        #[arg(short, long, default_value = "output.png")]
        output: PathBuf,

        /// Model type to use
        #[arg(short, long, value_enum, default_value = "sdxl")]
        model: ModelType,

        /// Number of inference steps
        #[arg(long, default_value = "30")]
        steps: usize,

        /// Guidance scale
        #[arg(long, default_value = "7.5")]
        guidance: f64,

        /// Path to vocabulary file (uses embedded CLIP vocab if not specified)
        #[arg(long)]
        vocab: Option<PathBuf>,

        /// Path to model weights directory or safetensors file
        #[arg(long)]
        weights: PathBuf,
    },

    /// Show information about available backends
    Info,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ModelType {
    /// Stable Diffusion 1.x (512x512 native)
    Sd1x,
    /// Stable Diffusion XL (1024x1024 native)
    Sdxl,
    /// SDXL with Refiner (highest quality)
    SdxlRefiner,
}

impl ModelType {
    /// Returns the native resolution (width, height) for this model
    fn native_resolution(&self) -> (usize, usize) {
        match self {
            ModelType::Sd1x => (512, 512),
            ModelType::Sdxl | ModelType::SdxlRefiner => (1024, 1024),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum SamplerType {
    /// DDIM - Deterministic, fast
    Ddim,
    /// DDPM - Original stochastic sampler
    Ddpm,
    /// Euler - Balanced speed/quality
    Euler,
    /// Euler Ancestral - Euler with noise injection
    EulerA,
    /// DPM++ 2M - Excellent quality (default, recommended)
    #[default]
    DpmPlusPlus,
    /// DPM++ 2M SDE - More detail, stochastic
    DpmPlusPlusSde,
    /// DPM++ 2S Ancestral
    Dpm2sA,
    /// DPM++ 3M SDE - Highest quality
    Dpm3mSde,
    /// Heun - High quality, 2x NFE
    Heun,
    /// LMS - Linear multi-step
    Lms,
    /// UniPC - Predictor-corrector
    UniPc,
    /// DEIS - Exponential integrator
    Deis,
    /// LCM - Fast (4-8 steps)
    Lcm,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum, PartialEq, Eq)]
enum ScheduleType {
    /// Simple/Normal - uniform spacing in timestep space
    Normal,
    /// Karras sigma schedule (default, better quality)
    #[default]
    Karras,
    /// Exponential spacing between sigma_max and sigma_min
    Exponential,
    /// SGM Uniform - uniform spacing in sigma space
    SgmUniform,
    /// Beta distribution spacing (more steps at high noise)
    Beta,
    /// Linear-Quadratic blend
    LinearQuadratic,
}

impl ScheduleType {
    /// Convert CLI schedule type to pipeline schedule type
    fn to_pipeline(self) -> burn_models::ScheduleType {
        match self {
            ScheduleType::Normal => burn_models::ScheduleType::Normal,
            ScheduleType::Karras => burn_models::ScheduleType::Karras,
            ScheduleType::Exponential => burn_models::ScheduleType::Exponential,
            ScheduleType::SgmUniform => burn_models::ScheduleType::SgmUniform,
            ScheduleType::Beta => burn_models::ScheduleType::Beta,
            ScheduleType::LinearQuadratic => burn_models::ScheduleType::LinearQuadratic,
        }
    }
}

impl SamplerType {
    /// Convert CLI sampler type to pipeline sampler type
    fn to_pipeline(self) -> burn_models::SamplerType {
        match self {
            SamplerType::Ddim => burn_models::SamplerType::Ddim,
            SamplerType::Ddpm => burn_models::SamplerType::Ddpm,
            SamplerType::Euler => burn_models::SamplerType::Euler,
            SamplerType::EulerA => burn_models::SamplerType::EulerA,
            SamplerType::DpmPlusPlus => burn_models::SamplerType::DpmPlusPlus,
            SamplerType::DpmPlusPlusSde => burn_models::SamplerType::DpmPlusPlusSde,
            SamplerType::Dpm2sA => burn_models::SamplerType::Dpm2sA,
            SamplerType::Dpm3mSde => burn_models::SamplerType::Dpm3mSde,
            SamplerType::Heun => burn_models::SamplerType::Heun,
            SamplerType::Lms => burn_models::SamplerType::Lms,
            SamplerType::UniPc => burn_models::SamplerType::UniPc,
            SamplerType::Deis => burn_models::SamplerType::Deis,
            SamplerType::Lcm => burn_models::SamplerType::Lcm,
        }
    }
}

/// Run LLM commands using the default backend
fn run_llm_command(command: llm::LlmCommands) -> Result<()> {
    // Use wgpu backend by default, fall back to ndarray
    #[cfg(feature = "wgpu")]
    {
        use burn_wgpu::{Wgpu, WgpuDevice};
        type Backend = Wgpu<f32>;
        let device = WgpuDevice::default();
        run_llm_command_with_backend::<Backend>(command, &device)
    }

    #[cfg(all(feature = "ndarray", not(feature = "wgpu")))]
    {
        use burn_ndarray::NdArray;
        type Backend = NdArray<f32>;
        let device = Default::default();
        run_llm_command_with_backend::<Backend>(command, &device)
    }

    #[cfg(not(any(feature = "wgpu", feature = "ndarray")))]
    {
        anyhow::bail!("No backend enabled. Enable 'wgpu' or 'ndarray' feature.")
    }
}

/// Run LLM commands with a specific backend
#[allow(unused_variables)]
fn run_llm_command_with_backend<B: burn::prelude::Backend>(
    command: llm::LlmCommands,
    device: &B::Device,
) -> Result<()> {
    match command {
        llm::LlmCommands::Generate {
            model,
            weights,
            prompt,
            max_tokens,
            temperature,
            top_p,
        } => llm::run_generate::<B>(
            model,
            weights,
            prompt,
            max_tokens,
            temperature,
            top_p,
            device,
        ),

        llm::LlmCommands::Chat {
            model,
            weights,
            system,
            max_tokens,
            temperature,
        } => llm::run_chat::<B>(model, weights, system, max_tokens, temperature, device),

        #[cfg(feature = "llm-serve")]
        llm::LlmCommands::Serve {
            model,
            weights,
            host,
            port,
        } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(llm::run_serve::<B>(model, weights, host, port, device))
        }
    }
}

/// Try to initialize CUDA and return true if successful
#[cfg(feature = "cuda")]
fn cuda_available() -> bool {
    use std::panic;
    // CudaDevice::default() will panic if CUDA is not available
    panic::catch_unwind(|| {
        let _ = burn_cuda::CudaDevice::default();
    })
    .is_ok()
}

/// Try to initialize WGPU and return true if successful
#[cfg(feature = "wgpu")]
fn wgpu_available() -> bool {
    use std::panic;
    // WgpuDevice::default() may panic if no GPU is available
    panic::catch_unwind(|| {
        let _ = burn_wgpu::WgpuDevice::default();
    })
    .is_ok()
}

/// Resolve Auto device to a concrete device
fn resolve_device(requested: Device) -> Device {
    match requested {
        Device::Auto => {
            // Try CUDA first, then WGPU, then CPU
            #[cfg(feature = "cuda")]
            if cuda_available() {
                eprintln!("[device] Auto-detected CUDA");
                return Device::Cuda;
            }
            #[cfg(feature = "wgpu")]
            if wgpu_available() {
                eprintln!("[device] Auto-detected WGPU");
                return Device::Wgpu;
            }
            #[cfg(feature = "cpu")]
            {
                eprintln!("[device] Falling back to CPU");
                return Device::Cpu;
            }
            #[allow(unreachable_code)]
            {
                panic!("No backend available. Enable 'cuda', 'wgpu', or 'cpu' feature.");
            }
        }
        other => other,
    }
}

/// Run SD 1.x generation with the specified backend
#[allow(clippy::too_many_arguments)]
fn run_sd1x_generate(
    prompt: &str,
    negative: &str,
    output: &PathBuf,
    vocab: Option<&PathBuf>,
    weights: &PathBuf,
    width: usize,
    height: usize,
    steps: usize,
    guidance: f64,
    _loras: &[PathBuf],
    _lora_scales: &[f64],
    precision: Precision,
    requested_device: Device,
    debug: &[String],
    flash_attention: bool,
    sampler: SamplerType,
    schedule: ScheduleType,
    denoise: f64,
) -> Result<()> {
    let device = resolve_device(requested_device);

    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda => {
            use burn_cuda::{Cuda, CudaDevice};
            let cuda_device = CudaDevice::default();

            match precision {
                #[cfg(feature = "precision-f16")]
                Precision::F16 => {
                    if flash_attention {
                        #[cfg(feature = "cubecl")]
                        {
                            use cubecl::cuda::CudaRuntime;
                            use half::f16;
                            run_sd1x_generate_flash::<CudaRuntime, f16, i32, u32>(
                                prompt,
                                negative,
                                output,
                                vocab,
                                weights,
                                width,
                                height,
                                steps,
                                guidance,
                                &cuda_device,
                                debug,
                                sampler,
                                schedule,
                                denoise,
                            )
                        }
                        #[cfg(not(feature = "cubecl"))]
                        {
                            anyhow::bail!(
                                "Flash attention requires the 'cubecl' feature.\n\n\
                                Rebuild with: cargo build --features cubecl\n\
                                Or disable flash attention: --flash-attention=false\n\
                                (Warning: f16 without flash attention may produce NaN)"
                            );
                        }
                    } else {
                        eprintln!(
                            "[warning] f16 precision without flash attention may produce NaN."
                        );
                        eprintln!(
                            "[warning] If output is garbled, use --flash-attention or --precision f32."
                        );
                        use half::f16;
                        type Backend = Cuda<f16>;
                        run_sd1x_generate_impl::<Backend>(
                            prompt,
                            negative,
                            output,
                            vocab,
                            weights,
                            width,
                            height,
                            steps,
                            guidance,
                            &cuda_device,
                            debug,
                            sampler,
                            schedule,
                            denoise,
                        )
                    }
                }
                #[cfg(feature = "precision-bf16")]
                Precision::Bf16 => {
                    if flash_attention {
                        #[cfg(feature = "cubecl")]
                        {
                            use cubecl::cuda::CudaRuntime;
                            use half::bf16;
                            run_sd1x_generate_flash::<CudaRuntime, bf16, i32, u32>(
                                prompt,
                                negative,
                                output,
                                vocab,
                                weights,
                                width,
                                height,
                                steps,
                                guidance,
                                &cuda_device,
                                debug,
                                sampler,
                                schedule,
                                denoise,
                            )
                        }
                        #[cfg(not(feature = "cubecl"))]
                        {
                            anyhow::bail!(
                                "Flash attention requires the 'cubecl' feature.\n\n\
                                Rebuild with: cargo build --features cubecl\n\
                                Or disable flash attention: --flash-attention=false"
                            );
                        }
                    } else {
                        use half::bf16;
                        type Backend = Cuda<bf16>;
                        run_sd1x_generate_impl::<Backend>(
                            prompt,
                            negative,
                            output,
                            vocab,
                            weights,
                            width,
                            height,
                            steps,
                            guidance,
                            &cuda_device,
                            debug,
                            sampler,
                            schedule,
                            denoise,
                        )
                    }
                }
                #[cfg(feature = "precision-f32")]
                Precision::F32 => {
                    if flash_attention {
                        #[cfg(feature = "cubecl")]
                        {
                            use cubecl::cuda::CudaRuntime;
                            run_sd1x_generate_flash::<CudaRuntime, f32, i32, u32>(
                                prompt,
                                negative,
                                output,
                                vocab,
                                weights,
                                width,
                                height,
                                steps,
                                guidance,
                                &cuda_device,
                                debug,
                                sampler,
                                schedule,
                                denoise,
                            )
                        }
                        #[cfg(not(feature = "cubecl"))]
                        {
                            anyhow::bail!(
                                "Flash attention requires the 'cubecl' feature.\n\
                                Rebuild with: cargo build --features cubecl"
                            );
                        }
                    } else {
                        type Backend = Cuda<f32>;
                        run_sd1x_generate_impl::<Backend>(
                            prompt,
                            negative,
                            output,
                            vocab,
                            weights,
                            width,
                            height,
                            steps,
                            guidance,
                            &cuda_device,
                            debug,
                            sampler,
                            schedule,
                            denoise,
                        )
                    }
                }
            }
        }

        #[cfg(feature = "wgpu")]
        Device::Wgpu => {
            use burn_wgpu::{Wgpu, WgpuDevice};
            let wgpu_device = WgpuDevice::default();

            match precision {
                #[cfg(feature = "precision-f16")]
                Precision::F16 => {
                    if flash_attention {
                        #[cfg(feature = "cubecl")]
                        {
                            use cubecl::wgpu::WgpuRuntime;
                            use half::f16;
                            run_sd1x_generate_flash::<WgpuRuntime, f16, i32, u32>(
                                prompt,
                                negative,
                                output,
                                vocab,
                                weights,
                                width,
                                height,
                                steps,
                                guidance,
                                &wgpu_device,
                                debug,
                                sampler,
                                schedule,
                                denoise,
                            )
                        }
                        #[cfg(not(feature = "cubecl"))]
                        {
                            anyhow::bail!(
                                "Flash attention requires the 'cubecl' feature.\n\n\
                                Rebuild with: cargo build --features cubecl\n\
                                Or disable flash attention: --flash-attention=false\n\
                                (Warning: f16 without flash attention may produce NaN)"
                            );
                        }
                    } else {
                        eprintln!(
                            "[warning] f16 precision without flash attention may produce NaN."
                        );
                        eprintln!(
                            "[warning] If output is garbled, use --flash-attention or --precision f32."
                        );
                        use half::f16;
                        type Backend = Wgpu<f16>;
                        run_sd1x_generate_impl::<Backend>(
                            prompt,
                            negative,
                            output,
                            vocab,
                            weights,
                            width,
                            height,
                            steps,
                            guidance,
                            &wgpu_device,
                            debug,
                            sampler,
                            schedule,
                            denoise,
                        )
                    }
                }
                #[cfg(feature = "precision-bf16")]
                Precision::Bf16 => {
                    if flash_attention {
                        #[cfg(feature = "cubecl")]
                        {
                            use cubecl::wgpu::WgpuRuntime;
                            use half::bf16;
                            run_sd1x_generate_flash::<WgpuRuntime, bf16, i32, u32>(
                                prompt,
                                negative,
                                output,
                                vocab,
                                weights,
                                width,
                                height,
                                steps,
                                guidance,
                                &wgpu_device,
                                debug,
                                sampler,
                                schedule,
                                denoise,
                            )
                        }
                        #[cfg(not(feature = "cubecl"))]
                        {
                            anyhow::bail!(
                                "Flash attention requires the 'cubecl' feature.\n\n\
                                Rebuild with: cargo build --features cubecl\n\
                                Or disable flash attention: --flash-attention=false"
                            );
                        }
                    } else {
                        use half::bf16;
                        type Backend = Wgpu<bf16>;
                        run_sd1x_generate_impl::<Backend>(
                            prompt,
                            negative,
                            output,
                            vocab,
                            weights,
                            width,
                            height,
                            steps,
                            guidance,
                            &wgpu_device,
                            debug,
                            sampler,
                            schedule,
                            denoise,
                        )
                    }
                }
                #[cfg(feature = "precision-f32")]
                Precision::F32 => {
                    type Backend = Wgpu<f32>;
                    run_sd1x_generate_impl::<Backend>(
                        prompt,
                        negative,
                        output,
                        vocab,
                        weights,
                        width,
                        height,
                        steps,
                        guidance,
                        &wgpu_device,
                        debug,
                        sampler,
                        schedule,
                        denoise,
                    )
                }
            }
        }

        #[cfg(feature = "cpu")]
        Device::Cpu => {
            use burn_cpu::{Cpu, CpuDevice};
            let cpu_device = CpuDevice;

            // Note: Flash attention is not available for CPU backend
            #[cfg(feature = "precision-f16")]
            if flash_attention && matches!(precision, Precision::F16) {
                eprintln!("[warning] Flash attention is not available for CPU backend.");
                eprintln!("[warning] f16 precision may produce NaN. Consider --precision f32.");
            }

            match precision {
                #[cfg(feature = "precision-f16")]
                Precision::F16 => {
                    use half::f16;
                    type Backend = Cpu<f16>;
                    run_sd1x_generate_impl::<Backend>(
                        prompt,
                        negative,
                        output,
                        vocab,
                        weights,
                        width,
                        height,
                        steps,
                        guidance,
                        &cpu_device,
                        debug,
                        sampler,
                        schedule,
                        denoise,
                    )
                }
                #[cfg(feature = "precision-bf16")]
                Precision::Bf16 => {
                    use half::bf16;
                    type Backend = Cpu<bf16>;
                    run_sd1x_generate_impl::<Backend>(
                        prompt,
                        negative,
                        output,
                        vocab,
                        weights,
                        width,
                        height,
                        steps,
                        guidance,
                        &cpu_device,
                        debug,
                        sampler,
                        schedule,
                        denoise,
                    )
                }
                #[cfg(feature = "precision-f32")]
                Precision::F32 => {
                    type Backend = Cpu<f32>;
                    run_sd1x_generate_impl::<Backend>(
                        prompt,
                        negative,
                        output,
                        vocab,
                        weights,
                        width,
                        height,
                        steps,
                        guidance,
                        &cpu_device,
                        debug,
                        sampler,
                        schedule,
                        denoise,
                    )
                }
            }
        }

        Device::Auto => unreachable!("Auto should be resolved above"),
    }
}

/// Debug flags parsed from --debug option
#[allow(dead_code)]
struct DebugFlags {
    timing: bool,
    shapes: bool,
    nan: bool,
    sampler: bool,
}

impl DebugFlags {
    fn from_args(debug: &[String]) -> Self {
        let all = debug.iter().any(|s| s == "all");
        Self {
            timing: all || debug.iter().any(|s| s == "timing"),
            shapes: all || debug.iter().any(|s| s == "shapes"),
            nan: all || debug.iter().any(|s| s == "nan"),
            sampler: all || debug.iter().any(|s| s == "sampler"),
        }
    }

    /// Convert to pipeline DebugConfig
    fn to_pipeline_config(&self) -> burn_models::DebugConfig {
        burn_models::DebugConfig {
            sampler: self.sampler,
            nan: self.nan,
        }
    }
}

/// SD 1.x generation implementation with a specific backend
#[allow(clippy::too_many_arguments)]
fn run_sd1x_generate_impl<B: Backend>(
    prompt: &str,
    negative: &str,
    output: &PathBuf,
    vocab: Option<&PathBuf>,
    weights: &PathBuf,
    width: usize,
    height: usize,
    steps: usize,
    guidance: f64,
    device: &B::Device,
    debug: &[String],
    sampler: SamplerType,
    schedule: ScheduleType,
    denoise: f64,
) -> Result<()> {
    use std::time::Instant;
    let debug_flags = DebugFlags::from_args(debug);
    let total_start = Instant::now();

    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}% {msg}")?
            .progress_chars("#>-"),
    );

    // Step 1: Load tokenizer (embedded vocab or from file)
    pb.set_message("Loading tokenizer...");
    pb.set_position(5);
    let start = Instant::now();
    let tokenizer = match vocab {
        Some(path) => ClipTokenizer::from_file(path).context("Failed to load vocabulary file")?,
        None => ClipTokenizer::new(), // Use embedded CLIP vocabulary
    };
    if debug_flags.timing {
        eprintln!("[timing] tokenizer: {:?}", start.elapsed());
    }

    // Step 2: Open weight loader
    pb.set_message("Opening weights...");
    pb.set_position(10);
    let start = Instant::now();
    let mut loader = SdWeightLoader::open(weights).context("Failed to open weights")?;
    if debug_flags.timing {
        eprintln!("[timing] open weights: {:?}", start.elapsed());
    }

    // Step 3: Load CLIP text encoder
    pb.set_message("Loading CLIP text encoder...");
    pb.set_position(15);
    let start = Instant::now();
    let clip_config = ClipConfig::sd1x();
    let text_encoder = loader
        .load_clip_text_encoder::<B>(&clip_config, device)
        .context("Failed to load CLIP text encoder")?;
    if debug_flags.timing {
        eprintln!("[timing] load CLIP: {:?}", start.elapsed());
    }

    // Step 4: Load UNet
    pb.set_message("Loading UNet...");
    pb.set_position(30);
    let start = Instant::now();
    let unet_config = UNetConfig::sd1x();
    let unet = loader
        .load_unet::<B>(&unet_config, device)
        .context("Failed to load UNet")?;
    if debug_flags.timing {
        eprintln!("[timing] load UNet: {:?}", start.elapsed());
    }

    // Step 5: Load VAE decoder
    pb.set_message("Loading VAE decoder...");
    pb.set_position(50);
    let start = Instant::now();
    let vae_config = DecoderConfig::sd();
    let vae_decoder = loader
        .load_vae_decoder::<B>(&vae_config, device)
        .context("Failed to load VAE decoder")?;
    if debug_flags.timing {
        eprintln!("[timing] load VAE: {:?}", start.elapsed());
    }

    // Step 6: Create pipeline
    pb.set_message("Initializing pipeline...");
    pb.set_position(55);

    // Log model shapes if requested
    if debug_flags.shapes {
        eprintln!("[shapes] VAE decoder:");
        eprintln!("  conv_in.weight: {:?}", vae_decoder.conv_in.weight.dims());
        eprintln!(
            "  mid_attn.q.weight: {:?}",
            vae_decoder.mid_attn.q.weight.dims()
        );
        eprintln!("  num_up_blocks: {}", vae_decoder.up_blocks.len());
        for (i, block) in vae_decoder.up_blocks.iter().enumerate() {
            eprintln!(
                "  up_block[{}]: {} res_blocks, upsample={:?}",
                i,
                block.res_blocks.len(),
                block.upsample.is_some()
            );
            if let Some(first_res) = block.res_blocks.first() {
                eprintln!(
                    "    first res_block.conv1.weight: {:?}",
                    first_res.conv1.weight.dims()
                );
            }
        }
        eprintln!(
            "  conv_out.weight: {:?}",
            vae_decoder.conv_out.weight.dims()
        );
        eprintln!(
            "  norm_out.weight: {:?}",
            vae_decoder.norm_out.weight.dims()
        );
    }

    let pipeline = burn_models::StableDiffusion1x {
        tokenizer,
        text_encoder,
        unet,
        vae_decoder,
        scheduler: NoiseSchedule::sd1x(device),
        device: device.clone(),
    };

    // Step 7: Generate image with per-step progress
    let start = Instant::now();
    let config = burn_models::SampleConfig {
        width,
        height,
        steps,
        guidance_scale: guidance,
        seed: None,
        sampler: sampler.to_pipeline(),
        schedule: schedule.to_pipeline(),
        denoise,
        debug: debug_flags.to_pipeline_config(),
    };

    // Track step timing for debug mode
    let mut step_start = Instant::now();

    let image_tensor = pipeline.generate_with_callback(
        prompt,
        negative,
        &config,
        burn_models::StepOutput::None, // No overhead - just progress
        |info| {
            // Update progress bar (60-90% range for generation)
            let progress = 60 + (info.step + 1) * 30 / info.total_steps;
            pb.set_position(progress as u64);
            pb.set_message(format!("Step {}/{}", info.step + 1, info.total_steps));

            if debug_flags.timing {
                eprintln!("[step {}] {:?}", info.step, step_start.elapsed());
                step_start = Instant::now();
            }
        },
    );

    if debug_flags.timing {
        eprintln!(
            "[timing] inference ({} steps): {:?}",
            steps,
            start.elapsed()
        );
    }

    // Step 8: Convert to image and save
    pb.set_message("Saving image...");
    pb.set_position(95);
    let start = Instant::now();

    // Debug: print image tensor info
    if debug_flags.sampler {
        eprintln!("[debug] Image tensor shape: {:?}", image_tensor.dims());
    }

    let rgb_data = burn_models::tensor_to_rgb(image_tensor.clone());
    let [_, _, h, w] = image_tensor.dims();

    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(w as u32, h as u32, rgb_data)
        .context("Failed to create image buffer")?;
    img.save(output)?;
    if debug_flags.timing {
        eprintln!("[timing] save image: {:?}", start.elapsed());
    }

    pb.finish_and_clear();
    let output_path = output
        .canonicalize()
        .unwrap_or_else(|_| output.to_path_buf());
    println!("\nSaved to: {}", output_path.display());

    if debug_flags.timing {
        eprintln!("[timing] total: {:?}", total_start.elapsed());
    }

    Ok(())
}

/// SD 1.x generation with Flash Attention for f16
///
/// This version converts the UNet to use flash attention, which uses f32
/// accumulation internally to prevent the NaN overflow that occurs with
/// standard attention in f16.
#[cfg(feature = "cubecl")]
#[allow(clippy::too_many_arguments)]
fn run_sd1x_generate_flash<R, F, I, BT>(
    prompt: &str,
    negative: &str,
    output: &PathBuf,
    vocab: Option<&PathBuf>,
    weights: &PathBuf,
    width: usize,
    height: usize,
    steps: usize,
    guidance: f64,
    device: &R::Device,
    debug: &[String],
    sampler: SamplerType,
    schedule: ScheduleType,
    _denoise: f64,
) -> Result<()>
where
    R: burn_cubecl::CubeRuntime,
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::BoolElement,
{
    use burn_cubecl::CubeBackend;
    use burn_models_samplers::{
        DdimConfig, DdimSampler, DpmConfig, DpmPlusPlusSampler, NoiseSchedule,
    };
    use burn_models_unet::cubecl::convert_unet;
    use std::time::Instant;

    let debug_flags = DebugFlags::from_args(debug);
    let total_start = Instant::now();

    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}% {msg}")?
            .progress_chars("#>-"),
    );

    // Step 1: Load tokenizer
    pb.set_message("Loading tokenizer...");
    pb.set_position(5);
    let start = Instant::now();
    let tokenizer = match vocab {
        Some(path) => ClipTokenizer::from_file(path).context("Failed to load vocabulary file")?,
        None => ClipTokenizer::new(),
    };
    if debug_flags.timing {
        eprintln!("[timing] tokenizer: {:?}", start.elapsed());
    }

    // Step 2: Open weight loader
    pb.set_message("Opening weights...");
    pb.set_position(10);
    let start = Instant::now();
    let mut loader = SdWeightLoader::open(weights).context("Failed to open weights")?;
    if debug_flags.timing {
        eprintln!("[timing] open weights: {:?}", start.elapsed());
    }

    // Step 3: Load CLIP text encoder
    pb.set_message("Loading CLIP text encoder...");
    pb.set_position(15);
    let start = Instant::now();
    let clip_config = ClipConfig::sd1x();
    let text_encoder = loader
        .load_clip_text_encoder::<CubeBackend<R, F, I, BT>>(&clip_config, device)
        .context("Failed to load CLIP text encoder")?;
    if debug_flags.timing {
        eprintln!("[timing] load CLIP: {:?}", start.elapsed());
    }

    // Step 4: Load UNet and convert to flash attention
    pb.set_message("Loading UNet (with flash attention)...");
    pb.set_position(30);
    let start = Instant::now();
    let unet_config = UNetConfig::sd1x();
    let unet_standard = loader
        .load_unet::<CubeBackend<R, F, I, BT>>(&unet_config, device)
        .context("Failed to load UNet")?;

    // Convert to flash attention version
    let unet = convert_unet(&unet_standard);
    drop(unet_standard); // Free memory from standard UNet
    if debug_flags.timing {
        eprintln!(
            "[timing] load UNet + convert to flash: {:?}",
            start.elapsed()
        );
    }
    eprintln!("[info] Flash attention enabled (f32 accumulation)");

    // Step 5: Load VAE decoder
    pb.set_message("Loading VAE decoder...");
    pb.set_position(50);
    let start = Instant::now();
    let vae_config = DecoderConfig::sd();
    let vae_decoder = loader
        .load_vae_decoder::<CubeBackend<R, F, I, BT>>(&vae_config, device)
        .context("Failed to load VAE decoder")?;
    if debug_flags.timing {
        eprintln!("[timing] load VAE: {:?}", start.elapsed());
    }

    // Step 6: Encode prompts
    pb.set_message("Encoding prompts...");
    pb.set_position(55);
    let start = Instant::now();

    // Tokenize and encode positive prompt
    let pos_tokens = tokenizer.encode_padded(prompt, 77);
    let pos_tensor: Tensor<CubeBackend<R, F, I, BT>, 1, Int> = Tensor::from_data(
        TensorData::new(
            pos_tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
            [77],
        ),
        device,
    );
    let cond = text_encoder.forward(pos_tensor.unsqueeze::<2>());

    // Tokenize and encode negative prompt
    let neg_tokens = tokenizer.encode_padded(negative, 77);
    let neg_tensor: Tensor<CubeBackend<R, F, I, BT>, 1, Int> = Tensor::from_data(
        TensorData::new(
            neg_tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
            [77],
        ),
        device,
    );
    let uncond = text_encoder.forward(neg_tensor.unsqueeze::<2>());

    if debug_flags.timing {
        eprintln!("[timing] encode prompts: {:?}", start.elapsed());
    }

    // Step 7: Sampling loop with flash attention UNet
    let start = Instant::now();
    let latent_height = height / 8;
    let latent_width = width / 8;
    let noise_schedule = NoiseSchedule::sd1x(device);
    let sigma_schedule = schedule.to_pipeline().to_sigma_schedule();

    // Run appropriate sampler based on config
    let latent: Tensor<CubeBackend<R, F, I, BT>, 4> = match sampler.to_pipeline() {
        burn_models::SamplerType::DpmPlusPlus => {
            // DPM++ 2M sampler
            let dpm_config = DpmConfig {
                num_inference_steps: steps,
                solver_order: 2,
                sigma_schedule,
                debug: debug_flags.sampler,
            };
            // Create a fresh schedule for DPM++
            let dpm_schedule = NoiseSchedule::sd1x(device);
            let mut dpm_sampler = DpmPlusPlusSampler::new(dpm_schedule, dpm_config, device);
            let timesteps_vec = dpm_sampler.timesteps().to_vec();
            let total_steps = dpm_sampler.num_steps();

            let sigmas = dpm_sampler.sigmas().to_vec();
            let mut latent = dpm_sampler.init_latent(1, 4, latent_height, latent_width, device);

            if debug_flags.timing {
                eprintln!("[info] DPM++ 2M sampler, init_sigma={:.4}", sigmas[0]);
            }

            let timestep_tensors: Vec<Tensor<CubeBackend<R, F, I, BT>, 1>> = timesteps_vec
                .iter()
                .map(|&t| {
                    Tensor::<CubeBackend<R, F, I, BT>, 1>::from_data(
                        TensorData::new(vec![t as f32], [1]),
                        device,
                    )
                })
                .collect();

            let mut step_start = Instant::now();

            for step_idx in 0..total_steps {
                let t = timestep_tensors[step_idx].clone();

                let noise_uncond = unet.forward(latent.clone(), t.clone(), uncond.clone());
                let noise_cond = unet.forward(latent.clone(), t, cond.clone());
                let noise_pred = noise_uncond.clone() + (noise_cond - noise_uncond) * guidance;

                latent = dpm_sampler.step(latent, noise_pred, step_idx);

                let progress = 60 + (step_idx + 1) * 30 / total_steps;
                pb.set_position(progress as u64);
                pb.set_message(format!("Step {}/{}", step_idx + 1, total_steps));

                if debug_flags.timing {
                    eprintln!(
                        "[step {}] t={} {:?}",
                        step_idx,
                        timesteps_vec[step_idx],
                        step_start.elapsed()
                    );
                    step_start = Instant::now();
                }
            }

            latent
        }
        _ => {
            // Default to DDIM for unsupported samplers
            if !matches!(sampler, SamplerType::Ddim) {
                eprintln!(
                    "[warning] Sampler {:?} not yet supported with flash attention, using DDIM",
                    sampler
                );
            }

            let ddim_config = DdimConfig {
                num_inference_steps: steps,
                eta: 0.0,
            };
            let ddim_sampler = DdimSampler::new(noise_schedule, ddim_config);
            let mut latent = ddim_sampler.init_latent(1, 4, latent_height, latent_width, device);

            let timestep_tensors: Vec<Tensor<CubeBackend<R, F, I, BT>, 1>> = ddim_sampler
                .timesteps()
                .iter()
                .map(|&t| {
                    Tensor::<CubeBackend<R, F, I, BT>, 1>::from_data(
                        TensorData::new(vec![t as f32], [1]),
                        device,
                    )
                })
                .collect();

            let timesteps_slice = ddim_sampler.timesteps();
            let total_steps = ddim_sampler.num_steps();
            let mut step_start = Instant::now();

            for step_idx in 0..total_steps {
                let t = timestep_tensors[step_idx].clone();

                let noise_uncond = unet.forward(latent.clone(), t.clone(), uncond.clone());
                let noise_cond = unet.forward(latent.clone(), t, cond.clone());

                if debug_flags.nan && step_idx == 0 {
                    let uncond_data = noise_uncond.clone().into_data();
                    let uncond_vec: Vec<f32> = uncond_data.to_vec().unwrap();
                    let min = uncond_vec.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = uncond_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let nan_count = uncond_vec.iter().filter(|x| x.is_nan()).count();
                    let inf_count = uncond_vec.iter().filter(|x| x.is_infinite()).count();
                    eprintln!(
                        "[debug] Step 0 noise_uncond: min={:.4}, max={:.4}, nan={}, inf={}",
                        min, max, nan_count, inf_count
                    );
                }

                let noise_pred = noise_uncond.clone() + (noise_cond - noise_uncond) * guidance;
                latent = ddim_sampler.step(latent, noise_pred, step_idx);

                if debug_flags.nan && step_idx == 0 {
                    let lat_data = latent.clone().into_data();
                    let lat_vec: Vec<f32> = lat_data.to_vec().unwrap();
                    let min = lat_vec.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = lat_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let nan_count = lat_vec.iter().filter(|x| x.is_nan()).count();
                    let inf_count = lat_vec.iter().filter(|x| x.is_infinite()).count();
                    eprintln!(
                        "[debug] Step 0 latent after: min={:.4}, max={:.4}, nan={}, inf={}",
                        min, max, nan_count, inf_count
                    );
                }

                let progress = 60 + (step_idx + 1) * 30 / total_steps;
                pb.set_position(progress as u64);
                pb.set_message(format!("Step {}/{}", step_idx + 1, total_steps));

                if debug_flags.timing {
                    eprintln!(
                        "[step {}] t={} {:?}",
                        step_idx,
                        timesteps_slice[step_idx],
                        step_start.elapsed()
                    );
                    step_start = Instant::now();
                }
            }

            latent
        }
    };

    if debug_flags.timing {
        eprintln!(
            "[timing] inference ({} steps): {:?}",
            steps,
            start.elapsed()
        );
    }

    // Step 8: Decode and save
    pb.set_message("Decoding to image...");
    pb.set_position(92);

    // Debug: print latent stats before VAE decode
    if debug_flags.nan {
        let latent_data = latent.clone().into_data().convert::<f32>();
        let latent_vec: Vec<f32> = latent_data.to_vec().unwrap();
        let min = latent_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = latent_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean: f32 = latent_vec.iter().sum::<f32>() / latent_vec.len() as f32;
        let nan_count = latent_vec.iter().filter(|x| x.is_nan()).count();
        let inf_count = latent_vec.iter().filter(|x| x.is_infinite()).count();
        eprintln!(
            "[debug] Pre-VAE latent stats: min={:.4}, max={:.4}, mean={:.4}, nan={}, inf={}",
            min, max, mean, nan_count, inf_count
        );
    }

    let start = Instant::now();
    let image_tensor = vae_decoder.decode_to_image(latent);

    // Debug: print output stats after VAE decode
    if debug_flags.nan {
        let img_data = image_tensor.clone().into_data().convert::<f32>();
        let img_vec: Vec<f32> = img_data.to_vec().unwrap();
        let min = img_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = img_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean: f32 = img_vec.iter().sum::<f32>() / img_vec.len() as f32;
        let nan_count = img_vec.iter().filter(|x| x.is_nan()).count();
        let inf_count = img_vec.iter().filter(|x| x.is_infinite()).count();
        eprintln!(
            "[debug] Post-VAE image stats: min={:.4}, max={:.4}, mean={:.4}, nan={}, inf={}",
            min, max, mean, nan_count, inf_count
        );
    }
    if debug_flags.timing {
        eprintln!("[timing] VAE decode: {:?}", start.elapsed());
    }

    pb.set_message("Saving image...");
    pb.set_position(95);
    let start = Instant::now();
    let rgb_data = burn_models::tensor_to_rgb(image_tensor.clone());
    let [_, _, h, w] = image_tensor.dims();

    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(w as u32, h as u32, rgb_data)
        .context("Failed to create image buffer")?;
    img.save(output)?;
    if debug_flags.timing {
        eprintln!("[timing] save image: {:?}", start.elapsed());
    }

    pb.finish_and_clear();
    let output_path = output
        .canonicalize()
        .unwrap_or_else(|_| output.to_path_buf());
    println!("\nSaved to: {}", output_path.display());

    if debug_flags.timing {
        eprintln!("[timing] total: {:?}", total_start.elapsed());
    }

    Ok(())
}

/// Application entry point
fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Llm { command } => run_llm_command(command),

        Commands::Generate {
            prompt,
            negative,
            output,
            model,
            width,
            height,
            steps,
            guidance,
            seed: _seed,
            vocab,
            weights,
            loras,
            lora_scales,
            precision,
            device,
            debug,
            flash_attention,
            sampler,
            schedule,
            denoise,
        } => {
            // Derive width/height from model if not specified
            let (native_w, native_h) = model.native_resolution();
            let width = width.unwrap_or(native_w);
            let height = height.unwrap_or(native_h);

            println!("burn-models: Stable Diffusion in pure Rust\n");
            println!("Configuration:");
            println!("  Model:    {:?}", model);
            println!("  Size:     {}x{}", width, height);
            println!("  Steps:    {}", steps);
            println!("  Guidance: {}", guidance);
            println!("  Sampler:  {:?}", sampler);
            println!("  Schedule: {:?}", schedule);
            if denoise < 1.0 {
                println!("  Denoise:  {:.2}", denoise);
            }
            println!("  Precision: {:?}", precision);
            println!("  Device:   {:?}", device);
            println!(
                "  Flash Attention: {}",
                if flash_attention {
                    "enabled"
                } else {
                    "disabled"
                }
            );
            println!(
                "  Vocab:    {}",
                vocab
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "(embedded)".to_string())
            );
            println!("  Weights:  {}", weights.display());
            if !loras.is_empty() {
                println!("  LoRAs:");
                for (i, lora_path) in loras.iter().enumerate() {
                    let scale = lora_scales.get(i).copied().unwrap_or(1.0);
                    println!("    - {} (scale: {})", lora_path.display(), scale);
                }
            }
            println!();

            // Check if weights path exists
            if !weights.exists() {
                anyhow::bail!(
                    "Weights path does not exist: {}\n\n\
                    Please provide a path to model weights.\n\
                    Supported formats:\n\
                    - Directory with .safetensors files (HuggingFace format)\n\
                    - Single .safetensors file",
                    weights.display()
                );
            }

            // Dispatch to backend-specific implementation
            match model {
                ModelType::Sd1x => {
                    run_sd1x_generate(
                        &prompt,
                        &negative,
                        &output,
                        vocab.as_ref(),
                        &weights,
                        width,
                        height,
                        steps,
                        guidance,
                        &loras,
                        &lora_scales,
                        precision,
                        device,
                        &debug,
                        flash_attention,
                        sampler,
                        schedule,
                        denoise,
                    )?;
                }
                ModelType::Sdxl | ModelType::SdxlRefiner => {
                    anyhow::bail!(
                        "SDXL generation is not yet implemented.\n\n\
                        Currently only SD 1.x models are supported.\n\
                        Use --model sd1x with a Stable Diffusion 1.x model."
                    );
                }
            }

            Ok(())
        }

        Commands::Img2Img {
            input,
            prompt,
            negative: _,
            output,
            model,
            strength,
            steps,
            guidance: _,
            vocab: _,
            weights: _,
        } => {
            println!("img2img: {} -> {}", input.display(), output.display());
            println!("Model: {:?}, Prompt: {}", model, prompt);
            println!("Strength: {}, Steps: {}", strength, steps);
            println!("\nimg2img pipeline not yet fully implemented.");
            Ok(())
        }

        Commands::Inpaint {
            input,
            mask,
            prompt,
            negative: _,
            output,
            model,
            steps,
            guidance: _,
            vocab: _,
            weights: _,
        } => {
            println!(
                "inpaint: {} + {} -> {}",
                input.display(),
                mask.display(),
                output.display()
            );
            println!("Model: {:?}, Prompt: {}", model, prompt);
            println!("Steps: {}", steps);
            println!("\nInpainting pipeline not yet fully implemented.");
            Ok(())
        }

        Commands::Info => {
            println!("burn-models: Model inference in pure Rust\n");
            println!("Available backends:");

            #[cfg(feature = "ndarray")]
            println!("  - ndarray (CPU, enabled)");
            #[cfg(not(feature = "ndarray"))]
            println!("  - ndarray (CPU, not enabled)");

            #[cfg(feature = "tch")]
            println!("  - tch/libtorch (CPU/CUDA/MPS, enabled)");
            #[cfg(not(feature = "tch"))]
            println!("  - tch/libtorch (CPU/CUDA/MPS, not enabled)");

            #[cfg(feature = "wgpu")]
            println!("  - wgpu (WebGPU, enabled)");
            #[cfg(not(feature = "wgpu"))]
            println!("  - wgpu (WebGPU, not enabled)");

            #[cfg(feature = "cuda")]
            println!("  - cuda (NVIDIA CUDA, enabled)");
            #[cfg(not(feature = "cuda"))]
            println!("  - cuda (NVIDIA CUDA, not enabled)");

            println!("\nStable Diffusion models:");
            println!("  - Stable Diffusion 1.x (sd1x)");
            println!("  - Stable Diffusion XL (sdxl)");
            println!("  - SDXL + Refiner (sdxl-refiner)");

            println!("\nStable Diffusion pipelines:");
            println!("  - Text-to-image (generate)");
            println!("  - Image-to-image (img2img)");
            println!("  - Inpainting (inpaint)");

            println!("\nLLM models:");
            println!("  - LLaMA 2/3 (llama)");
            println!("  - Mistral 7B (mistral)");
            println!("  - Mixtral MoE (mixtral)");
            println!("  - Gemma 2 (gemma)");
            println!("  - Phi-2/3 (phi)");
            println!("  - Qwen 1.5/2 (qwen)");
            println!("  - DeepSeek (deepseek)");
            println!("  - RWKV-7 (rwkv)");
            println!("  - Mamba SSM (mamba)");
            println!("  - Jamba hybrid (jamba)");

            println!("\nLLM commands:");
            println!("  - burn-models llm generate - Text completion");
            println!("  - burn-models llm chat - Interactive chat");
            #[cfg(feature = "llm-serve")]
            println!("  - burn-models llm serve - OpenAI-compatible server");

            println!("\nSupported extensions:");
            println!("  - LoRA (Kohya and Diffusers formats)");

            Ok(())
        }
    }
}

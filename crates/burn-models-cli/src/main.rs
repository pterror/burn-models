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
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum Precision {
    /// 32-bit float (more VRAM, no JIT overhead)
    F32,
    /// 16-bit float (less VRAM, faster after JIT warmup)
    #[default]
    F16,
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

use burn_models::DiffusionPipeline;
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

        /// Debug modes (comma-separated): timing, shapes, all
        /// Examples: --debug  --debug timing  --debug shapes,timing
        #[arg(long, value_delimiter = ',', num_args = 0.., default_missing_value = "all")]
        debug: Vec<String>,
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
) -> Result<()> {
    let device = resolve_device(requested_device);

    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda => {
            use burn_cuda::{Cuda, CudaDevice};
            let cuda_device = CudaDevice::default();

            match precision {
                Precision::F16 => {
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
                    )
                }
                Precision::F32 => {
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
                    )
                }
            }
        }

        #[cfg(feature = "wgpu")]
        Device::Wgpu => {
            use burn_wgpu::{Wgpu, WgpuDevice};
            let wgpu_device = WgpuDevice::default();

            match precision {
                Precision::F16 => {
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
                    )
                }
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
                    )
                }
            }
        }

        #[cfg(feature = "cpu")]
        Device::Cpu => {
            use burn_cpu::{Cpu, CpuDevice};
            let cpu_device = CpuDevice::default();

            match precision {
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
                    )
                }
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
                    )
                }
            }
        }

        Device::Auto => unreachable!("Auto should be resolved above"),
    }
}

/// Debug flags parsed from --debug option
struct DebugFlags {
    timing: bool,
    shapes: bool,
}

impl DebugFlags {
    fn from_args(debug: &[String]) -> Self {
        let all = debug.iter().any(|s| s == "all");
        Self {
            timing: all || debug.iter().any(|s| s == "timing"),
            shapes: all || debug.iter().any(|s| s == "shapes"),
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
            println!("  Precision: {:?}", precision);
            println!("  Device:   {:?}", device);
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

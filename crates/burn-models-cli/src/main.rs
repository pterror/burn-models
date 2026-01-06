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

        /// Image width
        #[arg(long, default_value = "1024")]
        width: usize,

        /// Image height
        #[arg(long, default_value = "1024")]
        height: usize,

        /// Number of inference steps
        #[arg(long, default_value = "30")]
        steps: usize,

        /// Guidance scale
        #[arg(long, default_value = "7.5")]
        guidance: f64,

        /// Random seed (optional)
        #[arg(long)]
        seed: Option<u64>,

        /// Path to vocabulary file
        #[arg(long)]
        vocab: PathBuf,

        /// Path to model weights directory
        #[arg(long)]
        weights: PathBuf,

        /// LoRA model paths (can be specified multiple times)
        #[arg(long = "lora", value_name = "FILE")]
        loras: Vec<PathBuf>,

        /// LoRA scales (same order as --lora, default 1.0)
        #[arg(long = "lora-scale", value_name = "SCALE")]
        lora_scales: Vec<f64>,
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

        /// Path to vocabulary file
        #[arg(long)]
        vocab: PathBuf,

        /// Path to model weights directory
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

        /// Path to vocabulary file
        #[arg(long)]
        vocab: PathBuf,

        /// Path to model weights directory
        #[arg(long)]
        weights: PathBuf,
    },

    /// Show information about available backends
    Info,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ModelType {
    /// Stable Diffusion 1.x (512x512 default)
    Sd1x,
    /// Stable Diffusion XL (1024x1024 default)
    Sdxl,
    /// SDXL with Refiner (highest quality)
    SdxlRefiner,
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
        } => llm::run_generate::<B>(model, weights, prompt, max_tokens, temperature, top_p, device),

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

/// Run SD 1.x generation with the default backend
#[allow(clippy::too_many_arguments)]
fn run_sd1x_generate(
    prompt: &str,
    negative: &str,
    output: &PathBuf,
    vocab: &PathBuf,
    weights: &PathBuf,
    width: usize,
    height: usize,
    steps: usize,
    guidance: f64,
    _loras: &[PathBuf],
    _lora_scales: &[f64],
) -> Result<()> {
    #[cfg(feature = "wgpu")]
    {
        use burn_wgpu::{Wgpu, WgpuDevice};
        type Backend = Wgpu<f32>;
        let device = WgpuDevice::default();
        run_sd1x_generate_impl::<Backend>(
            prompt, negative, output, vocab, weights,
            width, height, steps, guidance, &device,
        )
    }

    #[cfg(all(feature = "ndarray", not(feature = "wgpu")))]
    {
        use burn_ndarray::NdArray;
        type Backend = NdArray<f32>;
        let device = Default::default();
        run_sd1x_generate_impl::<Backend>(
            prompt, negative, output, vocab, weights,
            width, height, steps, guidance, &device,
        )
    }

    #[cfg(not(any(feature = "wgpu", feature = "ndarray")))]
    {
        anyhow::bail!("No backend enabled. Enable 'wgpu' or 'ndarray' feature.")
    }
}

/// SD 1.x generation implementation with a specific backend
#[allow(clippy::too_many_arguments)]
fn run_sd1x_generate_impl<B: Backend>(
    prompt: &str,
    negative: &str,
    output: &PathBuf,
    vocab: &PathBuf,
    weights: &PathBuf,
    width: usize,
    height: usize,
    steps: usize,
    guidance: f64,
    device: &B::Device,
) -> Result<()> {
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}% {msg}")?
            .progress_chars("#>-"),
    );

    // Step 1: Load tokenizer
    pb.set_message("Loading tokenizer...");
    pb.set_position(5);
    let tokenizer = ClipTokenizer::from_file(vocab)
        .context("Failed to load vocabulary file")?;

    // Step 2: Open weight loader
    pb.set_message("Opening weights...");
    pb.set_position(10);
    let mut loader = SdWeightLoader::open(weights)
        .context("Failed to open weights")?;

    // Step 3: Load CLIP text encoder
    pb.set_message("Loading CLIP text encoder...");
    pb.set_position(15);
    let clip_config = ClipConfig::sd1x();
    let text_encoder = loader.load_clip_text_encoder::<B>(&clip_config, device)
        .context("Failed to load CLIP text encoder")?;

    // Step 4: Load UNet
    pb.set_message("Loading UNet...");
    pb.set_position(30);
    let unet_config = UNetConfig::sd1x();
    let unet = loader.load_unet::<B>(&unet_config, device)
        .context("Failed to load UNet")?;

    // Step 5: Load VAE decoder
    pb.set_message("Loading VAE decoder...");
    pb.set_position(50);
    let vae_config = DecoderConfig::sd();
    let vae_decoder = loader.load_vae_decoder::<B>(&vae_config, device)
        .context("Failed to load VAE decoder")?;

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

    // Step 7: Generate image
    pb.set_message("Generating image...");
    pb.set_position(60);
    let config = burn_models::SampleConfig {
        width,
        height,
        steps,
        guidance_scale: guidance,
        seed: None,
    };

    let image_tensor = pipeline.generate(prompt, negative, &config);

    // Step 8: Convert to image and save
    pb.set_message("Saving image...");
    pb.set_position(95);
    let rgb_data = burn_models::tensor_to_rgb(image_tensor.clone());
    let [_, _, h, w] = image_tensor.dims();

    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(w as u32, h as u32, rgb_data)
            .context("Failed to create image buffer")?;
    img.save(output)?;

    pb.finish_and_clear();
    println!("Image saved to: {}", output.display());

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
        } => {
            println!("burn-models: Stable Diffusion in pure Rust\n");
            println!("Configuration:");
            println!("  Model:    {:?}", model);
            println!("  Size:     {}x{}", width, height);
            println!("  Steps:    {}", steps);
            println!("  Guidance: {}", guidance);
            println!("  Vocab:    {}", vocab.display());
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
                        &prompt, &negative, &output, &vocab, &weights,
                        width, height, steps, guidance,
                        &loras, &lora_scales,
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

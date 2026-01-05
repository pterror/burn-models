# TODO

## Next Up

- [x] Rename project from burn-image to burn-models
- [x] Restructure crates for general model inference

## Completed (Image Generation)

- [x] Explore stable-diffusion-burn source to understand architecture
- [x] Explore stable-diffusion-xl-burn source for SDXL differences
- [x] Define crate structure (workspace with 7 crates)
- [x] Implement weight loading from safetensors
- [x] Implement CLIP tokenizer
- [x] Implement CLIP text encoder
- [x] Implement DDIM sampler
- [x] Implement VAE decoder
- [x] Implement UNet

## Roadmap

### Phase 1: Foundation

- [x] Weight loading infrastructure (safetensors → Burn tensors)
- [x] Model serialization (save/load converted weights)
- [x] Basic tensor ops (silu, groupnorm, layernorm)
- [x] Attention mechanism (qkv_attention, causal_mask)

### Phase 2: SD 1.x Components

- [x] CLIP text encoder
- [x] VAE decoder (latent → image)
- [x] UNet (diffusion backbone)
- [x] DDIM sampler

### Phase 3: SD 1.x Pipeline

- [x] Text-to-image generation
- [x] Classifier-free guidance
- [x] VAE encoder (for img2img)
- [x] Image-to-image pipeline

### Phase 4: SDXL Components

- [x] OpenCLIP text encoder (second encoder)
- [x] SDXL UNet (larger, different architecture)
- [x] SDXL VAE (scaling factors: 0.13025 vs 0.18215)
- [x] Refiner model

### Phase 5: SDXL Pipeline

- [x] SDXL text-to-image
- [x] SDXL img2img
- [x] Inpainting
- [x] Base + refiner workflow

### Phase 6: Polish

- [x] Multi-backend support (CUDA, Metal, WGPU)
- [x] Memory optimization
- [x] CLI interface
- [x] Documentation

## Backlog

### Samplers
- [x] DDIM
- [x] Euler
- [x] Euler Ancestral
- [x] DPM++ 2M
- [x] DPM++ SDE
- [x] Heun
- [x] HeunPP2
- [x] DPM 2
- [x] DPM 2 Ancestral
- [x] LMS
- [x] DPM++ 3M SDE
- [x] DDPM
- [x] LCM (Latent Consistency Model)
- [x] iPNDM
- [x] iPNDM-v
- [x] DEIS
- [x] UniPC
- [x] SA-Solver
- [x] Euler CFG++
- [x] Euler Ancestral CFG++
- [x] DPM Fast
- [x] DPM Adaptive
- [x] DPM++ 2S Ancestral
- [x] DPM++ 2S Ancestral CFG++
- [x] DPM++ 2M CFG++
- [x] DPM++ 2M SDE Heun
- [x] Res Multistep

### Model Extensions
- [x] LoRA support (Kohya, Diffusers formats)
- [x] ControlNet integration (SD 1.x, SDXL)
- [x] Textual inversion / embeddings
- [x] IP-Adapter
- [x] T2I-Adapter

### Performance
- [x] fp16/bf16 inference (precision config)
- [x] Flash Attention (tiled, memory-efficient, sliced)
- [x] xFormers-style memory efficient attention
- [x] Model offloading (sequential CPU/GPU)
- [x] Batch processing

### Future Architectures

#### UNet-based (similar to current)
- [x] Stable Diffusion 2.x - OpenCLIP ViT-H/14 text encoder, v-prediction support in samplers
- [ ] Stable Cascade - Würstchen architecture, latent cascade

#### DiT-based (new architecture needed)
- [x] Flux (Black Forest Labs) - Flow matching, DiT architecture, weight loading
- [x] Stable Diffusion 3 / 3.5 - MMDiT (Multimodal Diffusion Transformer), rectified flow
- [x] Z-Image (Alibaba) - Single-stream DiT (S3-DiT), 6B params, text+image tokens in single stream
- [x] Qwen-Image (Alibaba) - 20B MMDiT, Qwen2.5-VL text encoder, joint attention with pooled conditioning
- [x] PixArt-α/Σ - Efficient DiT, T5 text encoder, cross-attention to text
- [x] AuraFlow - Open source flow-based, MMDiT joint attention
- [x] Hunyuan-DiT - Bilingual DiT model, dual text encoder (CLIP + MT5), skip connections

### Video Generation

- [x] Wan 2.x (Alibaba) - DiT + MoE with sparse routing, 3D VAE, factorized spatial-temporal attention
- [x] CogVideoX - Open source video DiT, factorized spatial-temporal attention
- [x] Mochi - Genmo's open source video model, asymmetric DiT, factorized attention
- [x] LTX-Video - Lightricks video model, causal temporal attention

### Large Language Models

- [x] LLaMA 2/3 architecture (basic implementation)
- [x] LLaMA weight loading from safetensors
- [x] Mixtral 8x7B/8x22B (LLaMA + MoE)
- [x] Mistral 7B/Nemo - Sliding window attention
- [x] Qwen 2.5 - Alibaba's multilingual LLM (0.5B to 72B)
- [x] Gemma 2 - Google's open models (interleaved sliding/global attention, logit soft-capping, GeGLU)
- [x] Phi-3/3.5 - Microsoft's small models (fused QKV/gate-up, GQA)
- [x] DeepSeek V1/V2 - Multi-head Latent Attention (MLA) for compressed KV cache

### Shared Building Blocks (burn-models-core)

#### Layers
- [x] Transformer block (shared by DiT, LLM, encoders)
- [x] RoPE (Rotary Position Embedding)
- [x] RMSNorm
- [x] SwiGLU / GeGLU activations
- [x] Multi-head attention with GQA support
- [x] MoE (Mixture of Experts) routing
- [x] DiT components (AdaLayerNorm, DiTBlock, PatchEmbed)
- [x] Sliding window attention mask

#### Inference Optimization
- [x] KV cache for autoregressive models
- [x] Paged attention (block tables, memory management, scheduler)
- [x] Continuous batching (iteration-level scheduling, preemption)
- [x] Speculative decoding (draft/target verification, acceptance sampling)

#### Quantization
- [x] INT8 dynamic quantization (symmetric/asymmetric)
- [x] INT4 quantization (group-wise, GPTQ style)
- [x] FP8 support

#### Video-specific
- [x] 3D VAE (temporal compression)
- [x] Temporal attention layers
- [x] Frame interpolation

# TODO

## Next Up

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
- [ ] Stable Diffusion 2.x - OpenCLIP text encoder, 768/512 variants, v-prediction
- [ ] Stable Cascade - Würstchen architecture, latent cascade

#### DiT-based (new architecture needed)
- [ ] Flux (Black Forest Labs) - Flow matching, DiT architecture
- [ ] Stable Diffusion 3 / 3.5 - MMDiT (Multimodal Diffusion Transformer), rectified flow
- [ ] Z-Image (Alibaba) - Single-stream DiT (S3-DiT), 6B params, very fast
- [ ] Qwen-Image (Alibaba) - 20B MMDiT, Qwen2.5-VL text encoder, Apache 2.0
- [ ] PixArt-α/Σ - Efficient DiT, T5 text encoder
- [ ] AuraFlow - Open source flow-based
- [ ] Hunyuan-DiT - Bilingual DiT model

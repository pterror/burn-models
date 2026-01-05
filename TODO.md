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
- [ ] Euler CFG++
- [ ] Euler Ancestral CFG++
- [ ] Heun
- [ ] HeunPP2
- [ ] DPM 2
- [ ] DPM 2 Ancestral
- [ ] LMS
- [ ] DPM Fast
- [ ] DPM Adaptive
- [ ] DPM++ 2S Ancestral
- [ ] DPM++ 2S Ancestral CFG++
- [ ] DPM++ 2M CFG++
- [ ] DPM++ 2M SDE Heun
- [ ] DPM++ 3M SDE
- [ ] DDPM
- [ ] LCM (Latent Consistency Model)
- [ ] iPNDM
- [ ] iPNDM-v
- [ ] DEIS
- [ ] Res Multistep
- [ ] SA-Solver

### Model Extensions
- [ ] LoRA support
- [ ] ControlNet integration
- [ ] Textual inversion / embeddings
- [ ] IP-Adapter
- [ ] T2I-Adapter

### Performance
- [ ] fp16/bf16 inference
- [ ] Flash Attention
- [ ] xFormers-style memory efficient attention
- [ ] Model offloading (sequential CPU/GPU)
- [ ] Batch processing

# TODO

## Next Up

- [x] Explore stable-diffusion-burn source to understand architecture
- [x] Explore stable-diffusion-xl-burn source for SDXL differences
- [x] Define crate structure (workspace with 7 crates)
- [ ] Implement weight loading from safetensors
- [ ] Implement basic attention mechanism
- [ ] Implement CLIP tokenizer

## Roadmap

### Phase 1: Foundation

- [ ] Weight loading infrastructure (safetensors → Burn tensors)
- [ ] Model serialization (save/load converted weights)
- [ ] Basic tensor ops and attention mechanisms

### Phase 2: SD 1.x Components

- [ ] CLIP text encoder
- [ ] VAE decoder (latent → image)
- [ ] UNet (diffusion backbone)
- [ ] DDIM sampler

### Phase 3: SD 1.x Pipeline

- [ ] Text-to-image generation
- [ ] Classifier-free guidance
- [ ] VAE encoder (for img2img)
- [ ] Image-to-image pipeline

### Phase 4: SDXL Components

- [ ] OpenCLIP text encoder (second encoder)
- [ ] SDXL UNet (larger, different architecture)
- [ ] SDXL VAE
- [ ] Refiner model

### Phase 5: SDXL Pipeline

- [ ] SDXL text-to-image
- [ ] SDXL img2img
- [ ] Inpainting
- [ ] Base + refiner workflow

### Phase 6: Polish

- [ ] Multi-backend support (CUDA, Metal, WGPU)
- [ ] Memory optimization
- [ ] CLI interface
- [ ] Documentation

## Backlog

- Alternative samplers (DPM++, Euler, etc.)
- LoRA support
- ControlNet integration
- Textual inversion / embeddings
- fp16/bf16 inference

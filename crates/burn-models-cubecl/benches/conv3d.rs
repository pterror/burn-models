//! Benchmark Conv3d: CubeCL kernel vs tensor-ops im2col
//!
//! Run with:
//!   cargo bench -p burn-models-cubecl --bench conv3d

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn conv3d_benchmarks(_c: &mut Criterion) {
    // TODO: Implement benchmarks once kernel compiles
    //
    // Planned benchmarks:
    // 1. Small kernel (3x3x3) on small input (16x64x64)
    // 2. Large kernel (5x5x5) on medium input (16x128x128)
    // 3. Stride=2 downsampling
    // 4. Grouped convolution
    //
    // Compare:
    // - burn-models-cubecl::conv3d (CubeCL kernel)
    // - burn-models-core::Conv3d::forward (im2col + matmul)
    println!("Conv3d benchmarks not yet implemented");
}

criterion_group!(benches, conv3d_benchmarks);
criterion_main!(benches);

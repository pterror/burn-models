//! Correctness tests comparing CubeCL conv3d against reference im2col implementation (WGPU)
//!
//! These tests verify that the CubeCL kernel produces numerically equivalent
//! results to the reference implementation.
//!
//! # Running the tests
//!
//! These tests require a GPU and the `wgpu` feature:
//!
//! ```sh
//! cargo test -p burn-models-cubecl --features wgpu --test correctness -- --ignored
//! ```

#![cfg(feature = "wgpu")]

use burn::prelude::*;
use burn_cubecl::CubeBackend;
use burn_wgpu::{WgpuDevice, WgpuRuntime};
use burn_models_cubecl::{conv3d, Conv3dOptions};

// Use CubeBackend directly to avoid FusionTensor wrapper
// The 4th generic is BoolElement (u32 for wgpu)
type TestBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

/// Reference im2col-based 3D convolution (copied from burn-models-core/src/vae3d.rs)
/// This is the "known good" implementation we're testing against.
mod reference {
    use burn::prelude::*;

    /// Pad a 5D tensor [B, C, T, H, W] with zeros
    fn pad_5d<B: Backend>(x: Tensor<B, 5>, padding: [usize; 3]) -> Tensor<B, 5> {
        let [batch, channels, time, height, width] = x.dims();
        let [p_t, p_h, p_w] = padding;
        let device = x.device();

        let new_t = time + 2 * p_t;
        let new_h = height + 2 * p_h;
        let new_w = width + 2 * p_w;

        // Create padded tensor
        let mut padded = Tensor::zeros([batch, channels, new_t, new_h, new_w], &device);

        // Copy original data to center using slice_assign
        padded = padded.slice_assign(
            [
                0..batch,
                0..channels,
                p_t..p_t + time,
                p_h..p_h + height,
                p_w..p_w + width,
            ],
            x,
        );

        padded
    }

    /// Extract 3D patches from input (im2col for 3D convolution)
    ///
    /// Input: [batch, in_channels, T, H, W]
    /// Output: [batch, in_channels * k_t * k_h * k_w, out_t * out_h * out_w]
    fn im2col_3d<B: Backend>(
        x: Tensor<B, 5>,
        kernel_size: [usize; 3],
        stride: [usize; 3],
        out_size: [usize; 3],
    ) -> Tensor<B, 3> {
        let [batch, in_ch, _, _, _] = x.dims();
        let [k_t, k_h, k_w] = kernel_size;
        let [s_t, s_h, s_w] = stride;
        let [out_t, out_h, out_w] = out_size;

        let kernel_elements = k_t * k_h * k_w;
        let out_positions = out_t * out_h * out_w;
        let col_size = in_ch * kernel_elements;

        let device = x.device();

        // Collect all patches
        let mut patches = Vec::with_capacity(out_positions);

        for ot in 0..out_t {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let t_start = ot * s_t;
                    let h_start = oh * s_h;
                    let w_start = ow * s_w;

                    // Extract patch [batch, in_ch, k_t, k_h, k_w]
                    let patch = x.clone().slice([
                        0..batch,
                        0..in_ch,
                        t_start..t_start + k_t,
                        h_start..h_start + k_h,
                        w_start..w_start + k_w,
                    ]);

                    // Flatten to [batch, in_ch * k_t * k_h * k_w]
                    let patch_flat = patch.reshape([batch, col_size]);
                    // Add position dimension [batch, col_size, 1]
                    let patch_col = patch_flat.unsqueeze_dim::<3>(2);
                    patches.push(patch_col);
                }
            }
        }

        if patches.is_empty() {
            Tensor::zeros([batch, col_size, 0], &device)
        } else {
            // Concatenate along position dimension: [batch, col_size, out_positions]
            Tensor::cat(patches, 2)
        }
    }

    /// Reference 3D convolution using im2col approach
    ///
    /// # Arguments
    /// * `input` - [batch, in_channels, T, H, W]
    /// * `weight` - [out_channels, in_channels, k_t, k_h, k_w]
    /// * `bias` - [out_channels] (optional)
    /// * `stride` - [stride_t, stride_h, stride_w]
    /// * `padding` - [pad_t, pad_h, pad_w]
    /// * `dilation` - [dilation_t, dilation_h, dilation_w] (must be [1,1,1] for now)
    pub fn conv3d_reference<B: Backend>(
        input: Tensor<B, 5>,
        weight: Tensor<B, 5>,
        bias: Option<Tensor<B, 1>>,
        stride: [usize; 3],
        padding: [usize; 3],
        _dilation: [usize; 3],
    ) -> Tensor<B, 5> {
        let [batch, _in_ch, _time, _height, _width] = input.dims();
        let [out_ch, in_ch_per_group, k_t, k_h, k_w] = weight.dims();

        // Reshape weight from [out_ch, in_ch, k_t, k_h, k_w] to [out_ch, in_ch * k_t * k_h * k_w]
        let kernel_elements = k_t * k_h * k_w;
        let weight_2d = weight.reshape([out_ch, in_ch_per_group * kernel_elements]);

        // Pad input if needed
        let x_padded = if padding[0] > 0 || padding[1] > 0 || padding[2] > 0 {
            pad_5d(input, padding)
        } else {
            input
        };
        let [_, _, pad_t, pad_h, pad_w] = x_padded.dims();

        // Calculate output dimensions
        let out_t = (pad_t - k_t) / stride[0] + 1;
        let out_h = (pad_h - k_h) / stride[1] + 1;
        let out_w = (pad_w - k_w) / stride[2] + 1;

        // im2col: extract patches and reshape to columns
        let cols = im2col_3d(x_padded, [k_t, k_h, k_w], stride, [out_t, out_h, out_w]);

        // Matrix multiply: weight @ cols
        // weight_2d is [out_ch, in_ch*k]
        // cols is [batch, in_ch*k, out_positions]
        let weight_expanded = weight_2d
            .unsqueeze_dim::<3>(0)
            .expand([batch, out_ch, in_ch_per_group * kernel_elements]);

        // Batched matmul: [batch, out_ch, in_ch*k] @ [batch, in_ch*k, out_positions] = [batch, out_ch, out_positions]
        let out = weight_expanded.matmul(cols);

        // Add bias if present
        let out = if let Some(bias) = bias {
            let bias_expanded = bias.reshape([1, out_ch, 1]);
            out + bias_expanded
        } else {
            out
        };

        // Reshape to [batch, out_ch, out_t, out_h, out_w]
        out.reshape([batch, out_ch, out_t, out_h, out_w])
    }
}

/// Convert Burn tensor to CubeTensor for the CubeCL kernel
fn to_cube_tensor<const D: usize>(
    tensor: Tensor<TestBackend, D>,
) -> burn_wgpu::CubeTensor<WgpuRuntime> {
    // into_primitive() returns TensorPrimitive::Float(CubeTensor<R>)
    match tensor.into_primitive() {
        burn::tensor::TensorPrimitive::Float(t) => t,
        _ => panic!("Expected float tensor"),
    }
}

/// Convert CubeTensor back to Burn tensor
fn from_cube_tensor<const D: usize>(
    tensor: burn_wgpu::CubeTensor<WgpuRuntime>,
) -> Tensor<TestBackend, D> {
    Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(tensor))
}

/// Run CubeCL conv3d kernel
fn run_cubecl_conv3d(
    input: Tensor<TestBackend, 5>,
    weight: Tensor<TestBackend, 5>,
    bias: Option<Tensor<TestBackend, 1>>,
    options: Conv3dOptions,
) -> Tensor<TestBackend, 5> {
    let input_cube = to_cube_tensor(input);
    let weight_cube = to_cube_tensor(weight);
    let bias_cube = bias.map(|b| to_cube_tensor(b));

    let output_cube = conv3d::<WgpuRuntime>(input_cube, weight_cube, bias_cube, options)
        .expect("CubeCL conv3d failed");

    from_cube_tensor(output_cube)
}

/// Compare two tensors for approximate equality
fn assert_tensors_approx_eq<const D: usize>(
    actual: Tensor<TestBackend, D>,
    expected: Tensor<TestBackend, D>,
    tolerance: f32,
    test_name: &str,
) {
    let actual_data: Vec<f32> = actual.into_data().to_vec().unwrap();
    let expected_data: Vec<f32> = expected.into_data().to_vec().unwrap();

    assert_eq!(
        actual_data.len(),
        expected_data.len(),
        "{}: tensor sizes don't match",
        test_name
    );

    let mut max_diff: f32 = 0.0;
    let mut max_diff_idx = 0;

    for (i, (a, e)) in actual_data.iter().zip(expected_data.iter()).enumerate() {
        let diff = (a - e).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    assert!(
        max_diff <= tolerance,
        "{}: max difference {} at index {} exceeds tolerance {} (actual={}, expected={})",
        test_name,
        max_diff,
        max_diff_idx,
        tolerance,
        actual_data[max_diff_idx],
        expected_data[max_diff_idx]
    );
}

#[test]
#[ignore = "requires GPU"]
fn test_conv3d_1x1x1_kernel() {
    // Simplest case: 1x1x1 kernel (effectively a 1x1 conv in all dimensions)
    let device = WgpuDevice::default();

    let batch = 1;
    let in_ch = 2;
    let out_ch = 3;
    let t = 4;
    let h = 4;
    let w = 4;

    let input = Tensor::<TestBackend, 5>::random(
        [batch, in_ch, t, h, w],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let weight = Tensor::<TestBackend, 5>::random(
        [out_ch, in_ch, 1, 1, 1],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &device,
    );

    let bias = Tensor::<TestBackend, 1>::random(
        [out_ch],
        burn::tensor::Distribution::Uniform(-0.1, 0.1),
        &device,
    );

    let options = Conv3dOptions {
        stride: [1, 1, 1],
        padding: [0, 0, 0],
        dilation: [1, 1, 1],
        groups: 1,
    };

    let cubecl_output = run_cubecl_conv3d(
        input.clone(),
        weight.clone(),
        Some(bias.clone()),
        options,
    );

    let reference_output = reference::conv3d_reference(
        input,
        weight,
        Some(bias),
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
    );

    assert_eq!(cubecl_output.dims(), reference_output.dims());
    assert_tensors_approx_eq(cubecl_output, reference_output, 1e-5, "1x1x1 kernel");
}

#[test]
#[ignore = "requires GPU"]
fn test_conv3d_3x3x3_kernel_no_padding() {
    // 3x3x3 kernel without padding
    let device = WgpuDevice::default();

    let batch = 1;
    let in_ch = 2;
    let out_ch = 4;
    let t = 6;
    let h = 6;
    let w = 6;

    let input = Tensor::<TestBackend, 5>::random(
        [batch, in_ch, t, h, w],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let weight = Tensor::<TestBackend, 5>::random(
        [out_ch, in_ch, 3, 3, 3],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &device,
    );

    let options = Conv3dOptions {
        stride: [1, 1, 1],
        padding: [0, 0, 0],
        dilation: [1, 1, 1],
        groups: 1,
    };

    let cubecl_output = run_cubecl_conv3d(
        input.clone(),
        weight.clone(),
        None,
        options,
    );

    let reference_output = reference::conv3d_reference(
        input,
        weight,
        None,
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
    );

    // Expected output: (6-3)/1 + 1 = 4 in each spatial dimension
    assert_eq!(cubecl_output.dims(), [1, 4, 4, 4, 4]);
    assert_eq!(reference_output.dims(), [1, 4, 4, 4, 4]);
    assert_tensors_approx_eq(cubecl_output, reference_output, 1e-4, "3x3x3 no padding");
}

#[test]
#[ignore = "requires GPU"]
fn test_conv3d_3x3x3_kernel_same_padding() {
    // 3x3x3 kernel with "same" padding
    let device = WgpuDevice::default();

    let batch = 1;
    let in_ch = 3;
    let out_ch = 8;
    let t = 8;
    let h = 8;
    let w = 8;

    let input = Tensor::<TestBackend, 5>::random(
        [batch, in_ch, t, h, w],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let weight = Tensor::<TestBackend, 5>::random(
        [out_ch, in_ch, 3, 3, 3],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &device,
    );

    let bias = Tensor::<TestBackend, 1>::random(
        [out_ch],
        burn::tensor::Distribution::Uniform(-0.1, 0.1),
        &device,
    );

    let options = Conv3dOptions {
        stride: [1, 1, 1],
        padding: [1, 1, 1], // Same padding
        dilation: [1, 1, 1],
        groups: 1,
    };

    let cubecl_output = run_cubecl_conv3d(
        input.clone(),
        weight.clone(),
        Some(bias.clone()),
        options,
    );

    let reference_output = reference::conv3d_reference(
        input,
        weight,
        Some(bias),
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    );

    // With same padding, output shape equals input shape
    assert_eq!(cubecl_output.dims(), [1, 8, 8, 8, 8]);
    assert_eq!(reference_output.dims(), [1, 8, 8, 8, 8]);
    assert_tensors_approx_eq(cubecl_output, reference_output, 1e-4, "3x3x3 same padding");
}

#[test]
#[ignore = "requires GPU"]
fn test_conv3d_stride_2() {
    // Test with stride 2 (downsampling)
    let device = WgpuDevice::default();

    let batch = 1;
    let in_ch = 2;
    let out_ch = 4;
    let t = 8;
    let h = 8;
    let w = 8;

    let input = Tensor::<TestBackend, 5>::random(
        [batch, in_ch, t, h, w],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let weight = Tensor::<TestBackend, 5>::random(
        [out_ch, in_ch, 3, 3, 3],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &device,
    );

    let options = Conv3dOptions {
        stride: [2, 2, 2],
        padding: [1, 1, 1],
        dilation: [1, 1, 1],
        groups: 1,
    };

    let cubecl_output = run_cubecl_conv3d(
        input.clone(),
        weight.clone(),
        None,
        options,
    );

    let reference_output = reference::conv3d_reference(
        input,
        weight,
        None,
        [2, 2, 2],
        [1, 1, 1],
        [1, 1, 1],
    );

    // Expected output: (8 + 2*1 - 3) / 2 + 1 = 4 in each dimension
    assert_eq!(cubecl_output.dims(), [1, 4, 4, 4, 4]);
    assert_eq!(reference_output.dims(), [1, 4, 4, 4, 4]);
    assert_tensors_approx_eq(cubecl_output, reference_output, 1e-4, "stride 2");
}

#[test]
#[ignore = "requires GPU"]
fn test_conv3d_batch_size_2() {
    // Test with batch size > 1
    let device = WgpuDevice::default();

    let batch = 2;
    let in_ch = 2;
    let out_ch = 3;
    let t = 4;
    let h = 4;
    let w = 4;

    let input = Tensor::<TestBackend, 5>::random(
        [batch, in_ch, t, h, w],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let weight = Tensor::<TestBackend, 5>::random(
        [out_ch, in_ch, 3, 3, 3],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &device,
    );

    let bias = Tensor::<TestBackend, 1>::random(
        [out_ch],
        burn::tensor::Distribution::Uniform(-0.1, 0.1),
        &device,
    );

    let options = Conv3dOptions {
        stride: [1, 1, 1],
        padding: [1, 1, 1],
        dilation: [1, 1, 1],
        groups: 1,
    };

    let cubecl_output = run_cubecl_conv3d(
        input.clone(),
        weight.clone(),
        Some(bias.clone()),
        options,
    );

    let reference_output = reference::conv3d_reference(
        input,
        weight,
        Some(bias),
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    );

    assert_eq!(cubecl_output.dims(), [2, 3, 4, 4, 4]);
    assert_eq!(reference_output.dims(), [2, 3, 4, 4, 4]);
    assert_tensors_approx_eq(cubecl_output, reference_output, 1e-4, "batch size 2");
}

#[test]
#[ignore = "requires GPU"]
fn test_conv3d_asymmetric() {
    // Test with asymmetric dimensions
    let device = WgpuDevice::default();

    let batch = 1;
    let in_ch = 3;
    let out_ch = 4;
    let t = 4;  // Time dimension smaller
    let h = 8;
    let w = 6;

    let input = Tensor::<TestBackend, 5>::random(
        [batch, in_ch, t, h, w],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    // Asymmetric kernel
    let weight = Tensor::<TestBackend, 5>::random(
        [out_ch, in_ch, 1, 3, 3],  // 1x3x3 kernel (no temporal extent)
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &device,
    );

    let options = Conv3dOptions {
        stride: [1, 1, 1],
        padding: [0, 1, 1],
        dilation: [1, 1, 1],
        groups: 1,
    };

    let cubecl_output = run_cubecl_conv3d(
        input.clone(),
        weight.clone(),
        None,
        options,
    );

    let reference_output = reference::conv3d_reference(
        input,
        weight,
        None,
        [1, 1, 1],
        [0, 1, 1],
        [1, 1, 1],
    );

    // Expected: T stays 4, H and W stay 8 and 6 with same padding
    assert_eq!(cubecl_output.dims(), [1, 4, 4, 8, 6]);
    assert_eq!(reference_output.dims(), [1, 4, 4, 8, 6]);
    assert_tensors_approx_eq(cubecl_output, reference_output, 1e-4, "asymmetric");
}

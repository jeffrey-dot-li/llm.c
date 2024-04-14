// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels (vector width), V = vocab_size

// reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
// both inp and out are (B,T,C) of the activations
// mean and rstd are (B,T) buffers, to be used later in backward pass
// at each position (b,t) of the input, the C-dimensional vector
// of activations gets normalized, then scaled and shifted
const EPS: f32 = 1e-5;

pub fn layernorm_forward(
    out: &mut [f32],
    mean_array: &mut [f32],
    rstd_array: &mut [f32],
    inp: &[f32],
    weights: &[f32],
    biases: &[f32],
    batch_size: usize,
    sequence_length: usize,
    channels: usize,
) {
    for b_idx in 0..batch_size {
        for t_idx in 0..sequence_length {
            // Seek to the input position inp[b,t,:]
            let x_offset_start = b_idx * sequence_length * channels + t_idx * channels;
            let x_offset_end = x_offset_start + channels;
            // calculate mean of layer
            let mean = inp[x_offset_start..x_offset_end].iter().sum::<f32>() / channels as f32;
            // calculate variance of layer
            let variance = inp[x_offset_start..x_offset_end]
                .iter()
                .fold(0.0, |acc, x| acc + (x - mean).powi(2))
                / channels as f32;

            // Calculate rstd (reciprocal standard deviation)
            let rstd = (variance + EPS).powf(-0.5);

            // Output offset is same as input offset.
            // TODO: Rewrite this to be immutable
            for i in 0..channels {
                let x_offset = x_offset_start + i;
                let normalized = (inp[x_offset] - mean) * rstd; // normalize input
                let output = normalized * weights[i] + biases[i]; // Fix this: weight and bias have length C????
                out[x_offset] = output;
            }
            mean_array[b_idx * sequence_length + t_idx] = mean;
            rstd_array[b_idx * sequence_length + t_idx] = rstd;
        }
    }
}

pub fn layernorm_backward(
    dinp: &mut [f32],
    dweight: &mut [f32],
    dbias: &mut [f32],
    dout: &mut [f32],
    input: &[f32],
    weights: &[f32],
    means: &[f32],
    rstds: &[f32],
    batch_size: usize,
    sequence_length: usize,
    channels: usize,
) {
    for batch_index in 0..batch_size {
        for t_index in 0..sequence_length {
            let batch_offset = batch_index * sequence_length + t_index;
            let sequence_offset = channels * batch_offset;
            let mean_bt = means[batch_offset];
            let rstd_bt = rstds[batch_offset];

            // First: two reduce operations
            let mut dnorm_mean: f32 = 0.0;
            let mut dnorm_norm_mean: f32 = 0.0;

            for i in 0..channels {
                let norm_bti = (input[i + sequence_offset] - mean_bt) * rstd_bt;
                let dnorm_i = weights[i] * dout[sequence_offset + i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / channels as f32;
            dnorm_norm_mean = dnorm_norm_mean / channels as f32;

            // now iterate again and accumulate all the gradients
            for i in 0..channels {
                let norm_bti = (input[i + sequence_offset] - mean_bt) * rstd_bt;
                let dnorm_i = weights[i] * dout[sequence_offset + i];
                // gradient contribution to bias
                dbias[i] += dout[sequence_offset + i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout[sequence_offset + i];
                // gradient contribution to input
                let mut dval: f32 = 0.0;
                dval += dnorm_i;
                dval -= dnorm_mean;
                dval -= norm_bti * dnorm_norm_mean;
                dval *= rstd_bt;
                dinp[i + sequence_offset] += dval;
            }
        }
    }
}

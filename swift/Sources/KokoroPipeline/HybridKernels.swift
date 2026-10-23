import Foundation

/// Metal kernels: masked AdaIN statistics, AdaIN affine + folded Snake, joins.
let hybridKernelSource = """
#include <metal_stdlib>
using namespace metal;

// One threadgroup per (channel c, row n): masked sum and sum of squares over w.
kernel void stats(device const half4 *x [[buffer(0)]], device const half *sm [[buffer(1)]],
                  device float2 *partial [[buffer(2)]], constant uint4 &dims [[buffer(3)]],
                  uint2 gid [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],
                  uint2 tc2 [[threads_per_threadgroup]]) {
    uint N = dims.x, C = dims.y, W = dims.z, W4 = W / 4;
    uint c = gid.x, n = gid.y, tid = tid2.x, tcount = tc2.x;
    // d > 1: the row is phase-major (sample q*d + p at p*Q + q). Q % 4 == 0 (plan
    // check), so each half4 holds one phase and its mask taps are d apart.
    uint d = dims.w, Q = W / d;
    device const half4 *row = x + (n * C + c) * W4;
    device const half *m = sm + n * W;
    // Shift by the row's first value so the second moment does not cancel
    // when |mean| >> std (some channels have mean ~3, std ~0.08).
    float k = row[0].x;
    float s = 0, q = 0;
    for (uint i4 = tid; i4 < W4; i4 += tcount) {
        float4 v = float4(row[i4]) - k, mk;
        if (d == 1) {
            mk = float4(((device const half4 *)m)[i4]);
        } else {
            uint i = i4 * 4, w = (i % Q) * d + i / Q;
            mk = float4(m[w], m[w + d], m[w + 2 * d], m[w + 3 * d]);
        }
        s += dot(v, mk); q += dot(v * v, mk);
    }
    float2 acc = simd_sum(float2(s, q));
    threadgroup float2 sh[32];
    if (simd_is_first()) sh[tid / 32] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float2 t = 0;
        for (uint i = 0; i < (tcount + 31) / 32; i++) t += sh[i];
        partial[(c * N + n) * 2] = t;
        partial[(c * N + n) * 2 + 1] = float2(k, 0);
    }
}

// `stats` fused with the residual add the ANE no longer does (--resid-gpu):
// x = half(y + r) is written over y (natural layout), and the statistics are
// taken of those fp16 values, which the apply pass then reads. Saves the
// separate join pass and one read of x. Same grid as `stats` (d must be 1).
kernel void stats_add(device half4 *x [[buffer(0)]], device const half *sm [[buffer(1)]],
                      device float2 *partial [[buffer(2)]], constant uint4 &dims [[buffer(3)]],
                      device const half4 *r [[buffer(4)]],
                      uint2 gid [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],
                      uint2 tc2 [[threads_per_threadgroup]]) {
    uint N = dims.x, C = dims.y, W = dims.z, W4 = W / 4;
    uint c = gid.x, n = gid.y, tid = tid2.x, tcount = tc2.x;
    uint base = (n * C + c) * W4;
    device const half4 *m = (device const half4 *)(sm + n * W);
    // One shift for the whole row, read before any thread overwrites x[base].
    threadgroup float shift;
    if (tid == 0) shift = float(half(float(x[base].x) + float(r[base].x)));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k = shift;
    float s = 0, q = 0;
    for (uint i4 = tid; i4 < W4; i4 += tcount) {
        half4 sum = half4(float4(x[base + i4]) + float4(r[base + i4]));
        x[base + i4] = sum;
        float4 v = float4(sum) - k, mk = float4(m[i4]);
        s += dot(v, mk); q += dot(v * v, mk);
    }
    float2 acc = simd_sum(float2(s, q));
    threadgroup float2 sh[32];
    if (simd_is_first()) sh[tid / 32] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float2 t = 0;
        for (uint i = 0; i < (tcount + 31) / 32; i++) t += sh[i];
        partial[(c * N + n) * 2] = t;
        partial[(c * N + n) * 2 + 1] = float2(k, 0);
    }
}

// One thread per channel: combine the rows' shifted sums (Chan et al.), then
// scale = g * rsqrt(var + eps), shift = b - mean * scale, where g, b already
// carry the style projection and the Snake alpha.
kernel void finalize(device const float2 *partial [[buffer(0)]], device const float2 *gb [[buffer(1)]],
                     device float2 *ss [[buffer(2)]], constant uint4 &dims [[buffer(3)]],
                     constant float4 &params [[buffer(4)]], uint c [[thread_position_in_grid]]) {
    uint N = dims.x, C = dims.y;
    if (c >= C) return;
    float count = params.x, eps = params.y;
    // gb holds C (g, b) pairs followed by the N per-row stats-mask counts.
    device const float *rowCount = (device const float *)(gb + C);
    float total = 0, total_q = 0;
    float ref = partial[(c * N) * 2 + 1].x;
    for (uint n = 0; n < N; n++) {
        float2 sq = partial[(c * N + n) * 2];
        float kn = partial[(c * N + n) * 2 + 1].x;
        float cn = rowCount[n];
        // Re-express row sums around the common reference `ref`.
        float d = kn - ref;
        total += sq.x + d * cn;
        total_q += sq.y + 2 * d * sq.x + d * d * cn;
    }
    float mean_r = total / count;                  // mean - ref
    float var = max(total_q / count - mean_r * mean_r, 0.0f);
    float2 g = gb[c];
    float scale = g.x * rsqrt(var + eps);
    ss[c] = float2(scale, g.y - (mean_r + ref) * scale);
}

// AdaIN affine + folded Snake between natural and phase-major layouts
// (natural w = q*D + p sits at phase position p*Q + q, Q = W/D). A threadgroup
// of 64 threads owns phase columns [q0, q0 + 256) of one row, i.e. 64*D natural
// half4s. Device memory is touched only as half4, contiguous across threads;
// the stride-D shuffle happens in threadgroup memory. (A float[20] block per
// thread with lane-strided half4 reads ran ~4x slower than `apply` on an A14;
// scalar-half gathers and scalar tiles were slower still.)
// Threadgroups (ceil(Q/256), C, N) of 64; pd = (dIn, dOut), one of them 1.
kernel void apply_phase(device const half4 *x [[buffer(0)]], device const half4 *zm [[buffer(1)]],
                        device const float2 *ss [[buffer(2)]], device half4 *out [[buffer(3)]],
                        constant uint4 &dims [[buffer(4)]], constant uint2 &pd [[buffer(5)]],
                        uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
    threadgroup float tile[256 * 5];
    uint C = dims.y, W = dims.z, c = C - 1 - tg.y, n = dims.x - 1 - tg.z;   // reversed, as in `apply`
    uint D = max(pd.x, pd.y), Q4 = W / D / 4, q04 = tg.x * 64;
    uint row4 = (n * C + c) * (W / 4), mrow4 = n * (W / 4);
    uint cols4 = min(64u, Q4 - q04);               // phase half4 columns in this tile
    uint nat4 = cols4 * D;                          // natural half4s in this tile
    uint natBase4 = q04 * D;                        // first natural half4 of the tile
    float2 sc = ss[c];
    if (pd.y > 1) {
        // natural -> phase: read natural half4s, apply, stage; write phase rows.
        for (uint k = t; k < nat4; k += 64) {
            float4 z = (float4(x[row4 + natBase4 + k]) * sc.x + sc.y) * float4(zm[mrow4 + natBase4 + k]);
            float4 sz = sin(z);
            float4 v = z + sz * sz;
            tile[4 * k] = v.x; tile[4 * k + 1] = v.y; tile[4 * k + 2] = v.z; tile[4 * k + 3] = v.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < cols4) {
            for (uint p = 0; p < D; p++) {
                uint b = 4 * t * D + p;             // natural offset of (q = q0 + 4t, p) in the tile
                out[row4 + p * Q4 + q04 + t] = half4(tile[b], tile[b + D], tile[b + 2 * D], tile[b + 3 * D]);
            }
        }
    } else {
        // phase -> natural: read phase rows into natural order; apply on write.
        if (t < cols4) {
            for (uint p = 0; p < D; p++) {
                float4 v = float4(x[row4 + p * Q4 + q04 + t]);
                uint b = 4 * t * D + p;
                tile[b] = v.x; tile[b + D] = v.y; tile[b + 2 * D] = v.z; tile[b + 3 * D] = v.w;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = t; k < nat4; k += 64) {
            float4 v = float4(tile[4 * k], tile[4 * k + 1], tile[4 * k + 2], tile[4 * k + 3]);
            float4 z = (v * sc.x + sc.y) * float4(zm[mrow4 + natBase4 + k]);
            float4 sz = sin(z);
            out[row4 + natBase4 + k] = half4(z + sz * sz);
        }
    }
}

kernel void apply(device const half4 *x [[buffer(0)]], device const half4 *zm [[buffer(1)]],
                  device const float2 *ss [[buffer(2)]], device half4 *out [[buffer(3)]],
                  constant uint4 &dims [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) {
    uint C = dims.y, W4 = dims.z / 4;
    if (gid.x >= W4) return;
    // Rows and channels in reverse: the stats pass just streamed this tensor
    // front to back, so its tail is still in the system cache.
    uint n = dims.x - 1 - gid.z, c = C - 1 - gid.y;
    uint idx = (n * C + c) * W4 + gid.x;
    float2 p = ss[c];
    float4 z = (float4(x[idx]) * p.x + p.y) * float4(zm[n * W4 + gid.x]);
    float4 sz = sin(z);
    out[idx] = half4(z + sz * sz);
}

// Polyphase upsampler output -> interleaved, plus noise branch, masked.
// poly (N, C*s, 1, Q): channel c*s + r holds sample q*s + r. Grid (W, C, N).
// p = (s, reflect, Q, offset); out column w is polyphase sample w + offset; with reflect, out[0] = y[1] and out[j] = y[j - 1],
// zero past the polyphase length (the ups1 reflect pad + zero extension).
kernel void unshuffle(device const half *poly [[buffer(0)]], device const half *xs [[buffer(1)]],
                      device const half *zm [[buffer(2)]], device half *out [[buffer(3)]],
                      constant uint4 &dims [[buffer(4)]], constant uint4 &p [[buffer(5)]],
                      uint3 gid [[thread_position_in_grid]]) {
    uint C = dims.y, W = dims.z, w = gid.x, c = gid.y, n = gid.z;
    if (w >= W) return;
    uint s = p.x, Q = p.z, wg = w + p.w;   // p.w: column offset of `out` in the polyphase row (stage-2 crop)
    int j = p.y ? (wg == 0 ? 1 : (int)wg - 1) : (int)wg;
    float v = 0;
    if (j < (int)(Q * s)) {
        uint q = (uint)j / s, r = (uint)j % s;
        v = poly[((n * C + c) * s + r) * Q + q];
    }
    uint idx = (n * C + c) * W + w;
    out[idx] = half((v + (float)xs[idx]) * (float)zm[n * W + w]);
}

// Grid (W/4, C, N). mode 0: out = a; 1: out = out + a; 2: out = (out + a) * zm * scale;
// 3: out = (a + b) * zm.
kernel void join(device const half4 *a [[buffer(0)]], device const half4 *b [[buffer(1)]],
                 device const half4 *zm [[buffer(2)]], device half4 *out [[buffer(3)]],
                 constant uint4 &dims [[buffer(4)]], constant float2 &mode [[buffer(5)]],
                 uint3 gid [[thread_position_in_grid]]) {
    uint C = dims.y, W4 = dims.z / 4;
    if (gid.x >= W4) return;
    uint idx = (gid.z * C + gid.y) * W4 + gid.x;
    float4 m = float4(zm[gid.z * W4 + gid.x]);
    int md = (int)mode.x;
    if (md == 0) out[idx] = a[idx];
    else if (md == 1) out[idx] = half4(float4(out[idx]) + float4(a[idx]));
    else if (md == 2) out[idx] = half4((float4(out[idx]) + float4(a[idx])) * m * mode.y);
    else if (md == 3) out[idx] = half4((float4(a[idx]) + float4(b[idx])) * m);
    // Modes 4-6 = modes 0-2 for a branch whose last residual b is still to add.
    else if (md == 4) out[idx] = half4(float4(a[idx]) + float4(b[idx]));
    else if (md == 5) out[idx] = half4(float4(out[idx]) + float4(a[idx]) + float4(b[idx]));
    else out[idx] = half4((float4(out[idx]) + float4(a[idx]) + float4(b[idx])) * m * mode.y);
}
"""

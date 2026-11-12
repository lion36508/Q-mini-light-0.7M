#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define QMINI_HAS_NEON 1
#else
#define QMINI_HAS_NEON 0
#endif

#if defined(_WIN32)
#define QMINI_EXPORT extern "C" __declspec(dllexport)
#else
#define QMINI_EXPORT extern "C" __attribute__((visibility("default")))
#endif

static inline int32_t qmini_decode_weight(
    const uint8_t* packed_row, int feature, int q2_version
) {
    const uint8_t packed = packed_row[feature / 4];
    const uint8_t code = static_cast<uint8_t>(
        (packed >> ((feature % 4) * 2)) & 0x03
    );
    if (q2_version == 1) {
        // Q2-v1 compatibility: doubled effective values {-4,-2,+2,+4}.
        switch (code) {
            case 0: return -4;
            case 1: return -2;
            case 2: return 2;
            default: return 4;
        }
    }
    // Q2-v2 normalized codebook, represented in half-units:
    // {-1,-0.5,+0.5,+1} -> {-2,-1,+1,+2}.
    switch (code) {
        case 0: return -2;
        case 1: return -1;
        case 2: return 1;
        default: return 2;
    }
}

static int64_t qmini_accumulate_i8_q2(
    const int8_t* input_row,
    const uint8_t* packed_row,
    int in_features,
    int q2_version
) {
#if QMINI_HAS_NEON
    // Expand eight 2-bit weights at a time, multiply signed INT8 x Q2 half-units,
    // widen products, and pairwise-accumulate in INT32 lanes. Flush each 512
    // input features into INT64 to keep the SIMD partial accumulator bounded.
    int64_t accumulator = 0;
    int32x4_t vector_sum = vdupq_n_s32(0);
    alignas(8) int8_t unpacked_weights[8];
    int vector_blocks = 0;
    int feature = 0;

    for (; feature + 8 <= in_features; feature += 8) {
        for (int lane = 0; lane < 8; ++lane) {
            unpacked_weights[lane] = static_cast<int8_t>(
                qmini_decode_weight(packed_row, feature + lane, q2_version)
            );
        }

        const int8x8_t activations = vld1_s8(input_row + feature);
        const int8x8_t weights = vld1_s8(unpacked_weights);
        const int16x8_t products = vmull_s8(activations, weights);
        vector_sum = vpadalq_s16(vector_sum, products);

        if (++vector_blocks == 64) {
            int32_t partial[4];
            vst1q_s32(partial, vector_sum);
            accumulator += static_cast<int64_t>(partial[0])
                         + static_cast<int64_t>(partial[1])
                         + static_cast<int64_t>(partial[2])
                         + static_cast<int64_t>(partial[3]);
            vector_sum = vdupq_n_s32(0);
            vector_blocks = 0;
        }
    }

    if (vector_blocks != 0) {
        int32_t partial[4];
        vst1q_s32(partial, vector_sum);
        accumulator += static_cast<int64_t>(partial[0])
                     + static_cast<int64_t>(partial[1])
                     + static_cast<int64_t>(partial[2])
                     + static_cast<int64_t>(partial[3]);
    }

    // Scalar tail for dimensions not divisible by eight.
    for (; feature < in_features; ++feature) {
        accumulator += static_cast<int64_t>(input_row[feature]) *
                       qmini_decode_weight(packed_row, feature, q2_version);
    }
    return accumulator;
#else
    int64_t accumulator = 0;
    for (int feature = 0; feature < in_features; ++feature) {
        accumulator += static_cast<int64_t>(input_row[feature]) *
                       qmini_decode_weight(packed_row, feature, q2_version);
    }
    return accumulator;
#endif
}

// Q2-v1 legacy symbols are {-2,-1,+1,+2}; Q2-v2 production symbols are
// {-1,-0.5,+0.5,+1}. Both use four symbols per byte and row scales.
// Integer accumulation uses doubled half-units and applies the 0.5 factor
// exactly once. ARM builds use NEON when the target supports it.
QMINI_EXPORT int qmini_linear_i8_v2(
    const int8_t* input,
    const uint8_t* packed_weights,
    const float* weight_scales,
    const float* input_scales,
    const float* bias,
    float* output,
    int rows,
    int in_features,
    int out_features,
    int q2_version
) {
    if (!input || !packed_weights || !weight_scales || !input_scales || !output) {
        return 1;
    }
    if (rows <= 0 || in_features <= 0 || out_features <= 0 ||
        in_features > std::numeric_limits<int>::max() - 3 ||
        (q2_version != 1 && q2_version != 2)) {
        return 2;
    }

    for (int row = 0; row < rows; ++row) {
        if (!std::isfinite(input_scales[row]) || input_scales[row] <= 0.0f) {
            return 3;
        }
    }
    for (int output_index = 0; output_index < out_features; ++output_index) {
        if (!std::isfinite(weight_scales[output_index]) ||
            (q2_version == 1
                ? weight_scales[output_index] <= 0.0f
                : weight_scales[output_index] < 0.0f)) {
            return 4;
        }
        if (bias && !std::isfinite(bias[output_index])) {
            return 5;
        }
    }

    const int bytes_per_row = (in_features + 3) / 4;
    for (int row = 0; row < rows; ++row) {
        for (int output_index = 0; output_index < out_features; ++output_index) {
            const uint8_t* weight_row =
                packed_weights + static_cast<std::size_t>(output_index) *
                static_cast<std::size_t>(bytes_per_row);
            const int8_t* input_row =
                input + static_cast<std::size_t>(row) *
                static_cast<std::size_t>(in_features);

            const int64_t accumulator =
                qmini_accumulate_i8_q2(input_row, weight_row, in_features, q2_version);

            const float offset = bias ? bias[output_index] : 0.0f;
            const float scaled = static_cast<float>(accumulator) *
                input_scales[row] * weight_scales[output_index] * 0.5f;
            output[static_cast<std::size_t>(row) *
                static_cast<std::size_t>(out_features) +
                static_cast<std::size_t>(output_index)] = scaled + offset;
        }
    }
    return 0;
}

QMINI_EXPORT int qmini_kernel_has_neon() {
    return QMINI_HAS_NEON;
}

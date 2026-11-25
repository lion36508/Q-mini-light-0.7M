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


static inline int qmini_octave_shift_for(int8_t coefficient) {
    switch (coefficient < 0 ? -coefficient : coefficient) {
        case 1: return 0;
        case 2: return 1;
        case 4: return 2;
        case 8: return 3;
        case 0: return 0;
        default: return -1;
    }
}

static inline int64_t qmini_octave_shift_add_term(
    int8_t activation, int8_t coefficient
) {
    if (activation == 0 || coefficient == 0) {
        return 0;
    }
    const int shift = qmini_octave_shift_for(coefficient);
    if (shift < 0) {
        return 0;
    }
    const int32_t signed_activation = static_cast<int32_t>(activation);
    const uint32_t magnitude = static_cast<uint32_t>(
        signed_activation < 0 ? -signed_activation : signed_activation
    ) << shift;
    const bool negative = (signed_activation < 0) != (coefficient < 0);
    const int64_t term = static_cast<int64_t>(magnitude);
    return negative ? -term : term;
}

struct QMiniOctaveDecodeTable {
    int8_t coefficients[256][3];

    QMiniOctaveDecodeTable() {
        for (int token = 0; token < 256; ++token) {
            for (int lane = 0; lane < 3; ++lane) {
                coefficients[token][lane] = static_cast<int8_t>(-128);
            }
        }
        static constexpr int8_t normal[5] = {-2, -1, 0, 1, 2};
        static constexpr int8_t boosted[5] = {-8, -4, 0, 4, 8};
        for (int token = 0; token < 250; ++token) {
            const int code = token >= 125 ? token - 125 : token;
            const int8_t* levels = token >= 125 ? boosted : normal;
            coefficients[token][0] = levels[code % 5];
            coefficients[token][1] = levels[(code / 5) % 5];
            coefficients[token][2] = levels[code / 25];
        }
    }
};

static inline const QMiniOctaveDecodeTable& qmini_octave_decode_table() {
    static const QMiniOctaveDecodeTable table;
    return table;
}

#if QMINI_HAS_NEON
static inline int64_t qmini_octave_accumulate_eight_neon(
    const int8_t* input,
    int feature_start,
    const int8_t* coefficients
) {
    int16_t shifts[8];
    uint16_t negative_mask[8];
    uint16_t zero_mask[8];
    for (int lane = 0; lane < 8; ++lane) {
        const int8_t coefficient = coefficients[lane];
        const int shift = qmini_octave_shift_for(coefficient);
        if (shift < 0) {
            return 0;
        }
        shifts[lane] = static_cast<int16_t>(shift);
        negative_mask[lane] = coefficient < 0 ? 0xFFFFu : 0u;
        zero_mask[lane] = coefficient == 0 ? 0xFFFFu : 0u;
    }

    // Eight INT8 activations are widened, shifted by 0..3, sign-selected,
    // and pairwise accumulated. No activation*weight multiply is performed.
    const int8x8_t activations_i8 = vld1_s8(input + feature_start);
    const int16x8_t activations_i16 = vmovl_s8(activations_i8);
    int16x8_t shifted = vshlq_s16(activations_i16, vld1q_s16(shifts));
    const int16x8_t negative = vnegq_s16(shifted);
    shifted = vbslq_s16(vld1q_u16(negative_mask), negative, shifted);
    shifted = vbslq_s16(vld1q_u16(zero_mask), vdupq_n_s16(0), shifted);
    const int32x4_t pair_sums = vpaddlq_s16(shifted);
    int32_t lanes[4];
    vst1q_s32(lanes, pair_sums);
    return static_cast<int64_t>(lanes[0])
         + static_cast<int64_t>(lanes[1])
         + static_cast<int64_t>(lanes[2])
         + static_cast<int64_t>(lanes[3]);
}
#endif

static inline int64_t qmini_octave_accumulate_block(
    const int8_t* input,
    int feature_start,
    const int8_t* coefficients,
    int count
) {
#if QMINI_HAS_NEON
    if (count == 8) {
        return qmini_octave_accumulate_eight_neon(
            input, feature_start, coefficients
        );
    }
#endif
    int64_t accumulator = 0;
    for (int lane = 0; lane < count; ++lane) {
        accumulator += qmini_octave_shift_add_term(
            input[feature_start + lane], coefficients[lane]
        );
    }
    return accumulator;
}

// Octave-Shift native projection. The packed stream stores one byte per
// triplet, with 250..254 skipping 4/8/16/32/64 all-zero triplet-groups and
// 255 followed by an extended skip count. Output remains in half-units;
// group/output scales are intentionally applied by a later requantization.
QMINI_EXPORT int qmini_octave_linear_i8_v1(
    const int8_t* input,
    const uint8_t* packed_stream,
    const uint32_t* row_offsets,
    const int64_t* bias_half_units,
    int64_t* output,
    int rows,
    int in_features,
    int out_features,
    int packed_size
) {
    if (!input || !packed_stream || !row_offsets || !output) {
        return 1;
    }
    if (rows <= 0 || in_features <= 0 || in_features > std::numeric_limits<int>::max() - 2 ||
        out_features <= 0 || packed_size < 0) {
        return 2;
    }
    if (row_offsets[0] != 0 || row_offsets[out_features] !=
        static_cast<uint32_t>(packed_size)) {
        return 3;
    }
    for (int output_index = 0; output_index < out_features; ++output_index) {
        if (row_offsets[output_index] > row_offsets[output_index + 1] ||
            row_offsets[output_index + 1] > static_cast<uint32_t>(packed_size)) {
            return 4;
        }
    }

    const int required_groups = (in_features + 2) / 3;
    const auto& table = qmini_octave_decode_table();
    for (int row = 0; row < rows; ++row) {
        const int8_t* input_row = input +
            static_cast<std::size_t>(row) * static_cast<std::size_t>(in_features);
        for (int output_index = 0; output_index < out_features; ++output_index) {
            std::size_t cursor = row_offsets[output_index];
            const std::size_t end = row_offsets[output_index + 1];
            int logical_group = 0;
            int logical_feature = 0;
            int64_t accumulator = 0;
            int buffered = 0;
            int buffer_start = 0;
            int8_t coefficient_buffer[8] = {0, 0, 0, 0, 0, 0, 0, 0};

            while (cursor < end) {
                const uint8_t token = packed_stream[cursor++];
                if (token >= 250) {
                    if (buffered != 0) {
                        accumulator += qmini_octave_accumulate_block(
                            input_row, buffer_start, coefficient_buffer, buffered
                        );
                        buffered = 0;
                    }

                    int skip = 0;
                    if (token == 255) {
                        if (cursor >= end) {
                            return 5;
                        }
                        skip = static_cast<int>(packed_stream[cursor++]);
                        if (skip < 1) {
                            return 6;
                        }
                    } else {
                        switch (token) {
                            case 250: skip = 4; break;
                            case 251: skip = 8; break;
                            case 252: skip = 16; break;
                            case 253: skip = 32; break;
                            case 254: skip = 64; break;
                            default: return 7;
                        }
                    }
                    if (logical_group > required_groups - skip) {
                        return 8;
                    }
                    logical_group += skip;
                    logical_feature += skip;
                    logical_feature += skip;
                    logical_feature += skip;
                    continue;
                }

                if (logical_group >= required_groups) {
                    return 9;
                }
                const int8_t* decoded = table.coefficients[token];
                for (int lane = 0; lane < 3; ++lane) {
                    if (logical_feature < in_features) {
                        if (buffered == 0) {
                            buffer_start = logical_feature;
                        }
                        coefficient_buffer[buffered++] = decoded[lane];
                        if (buffered == 8) {
                            accumulator += qmini_octave_accumulate_block(
                                input_row, buffer_start, coefficient_buffer, buffered
                            );
                            buffered = 0;
                        }
                    }
                    ++logical_feature;
                }
                ++logical_group;
            }

            if (buffered != 0) {
                accumulator += qmini_octave_accumulate_block(
                    input_row, buffer_start, coefficient_buffer, buffered
                );
            }
            if (logical_group != required_groups || cursor != end) {
                return 10;
            }
            if (bias_half_units) {
                accumulator += bias_half_units[output_index];
            }
            output[static_cast<std::size_t>(row) *
                static_cast<std::size_t>(out_features) +
                static_cast<std::size_t>(output_index)] = accumulator;
        }
    }
    return 0;
}

QMINI_EXPORT int qmini_kernel_has_neon() {
    return QMINI_HAS_NEON;
}

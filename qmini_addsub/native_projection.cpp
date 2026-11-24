#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define QMINI_ADDSUB_HAS_NEON 1
#else
#define QMINI_ADDSUB_HAS_NEON 0
#endif

#if defined(_WIN32)
#define QMINI_ADDSUB_EXPORT extern "C" __declspec(dllexport)
#else
#define QMINI_ADDSUB_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
using LookupBank = std::array<std::array<int16_t, 256>, 5>;

constexpr LookupBank make_contribution_lookup() {
    LookupBank bank{};
    int encoded = 0;
    while (encoded < 256) {
        const int value = encoded - 128;
        const int triple = value + value + value;
        bank[0][static_cast<std::size_t>(encoded)] = static_cast<int16_t>(-triple);
        bank[1][static_cast<std::size_t>(encoded)] = static_cast<int16_t>(-value);
        bank[2][static_cast<std::size_t>(encoded)] = 0;
        bank[3][static_cast<std::size_t>(encoded)] = static_cast<int16_t>(value);
        bank[4][static_cast<std::size_t>(encoded)] = static_cast<int16_t>(triple);
        ++encoded;
    }
    return bank;
}

constexpr LookupBank kContributionLookup = make_contribution_lookup();
constexpr int kMaxInputFeatures = 5000000;
}  // namespace

// Weight codes are 0..4 for {-1.5, -0.5, 0, +0.5, +1.5}.
// The weight matrix is stored feature-major: [in_features][out_features].
// Activations are signed int8 in [-127, 127]. Output accumulators are in doubled units.
// The numeric path uses lookup + integer addition only; no scale or bias is applied here.
QMINI_ADDSUB_EXPORT int qmini_addsub_projection_i8(
    const int8_t* activations,
    const uint8_t* weights_by_input,
    int64_t* output,
    int rows,
    int in_features,
    int out_features
) {
    if (activations == nullptr || weights_by_input == nullptr || output == nullptr) {
        return 1;
    }
    if (rows <= 0 || in_features <= 0 || out_features <= 0 ||
        in_features > kMaxInputFeatures) {
        return 2;
    }

    // Reject unsupported activation values and reserved weight symbols before writing output.
    const int8_t* activation_check = activations;
    for (int row = 0; row < rows; ++row) {
        for (int feature = 0; feature < in_features; ++feature) {
            const int value = static_cast<int>(activation_check[feature]);
            if (value < -127 || value > 127) {
                return 3;
            }
        }
        activation_check += in_features;
    }
    const uint8_t* weight_check = weights_by_input;
    for (int feature = 0; feature < in_features; ++feature) {
        for (int output_index = 0; output_index < out_features; ++output_index) {
            if (weight_check[output_index] > 4) {
                return 4;
            }
        }
        weight_check += out_features;
    }

    const int8_t* input_row = activations;
    int64_t* output_row = output;
    for (int row = 0; row < rows; ++row) {
        int output_base = 0;
#if QMINI_ADDSUB_HAS_NEON
        // Eight output channels are accumulated in parallel. Each lane uses a
        // comparison-selected contribution from the five-symbol lookup alphabet.
        for (; output_base + 8 <= out_features; output_base += 8) {
            int32x4_t sum_low = vdupq_n_s32(0);
            int32x4_t sum_high = vdupq_n_s32(0);
            const uint8_t* feature_codes = weights_by_input + output_base;
            for (int feature = 0; feature < in_features; ++feature) {
                const int16_t value = static_cast<int16_t>(input_row[feature]);
                const int16x8_t activation = vdupq_n_s16(value);
                const int16x8_t negative = vnegq_s16(activation);
                const int16x8_t triple = vaddq_s16(vaddq_s16(activation, activation), activation);
                const int16x8_t negative_triple = vnegq_s16(triple);
                const uint8x8_t codes = vld1_u8(feature_codes);
                int16x8_t contribution = vdupq_n_s16(0);
                contribution = vbslq_s16(vmovl_u8(vceq_u8(codes, vdup_n_u8(0))), negative_triple, contribution);
                contribution = vbslq_s16(vmovl_u8(vceq_u8(codes, vdup_n_u8(1))), negative, contribution);
                contribution = vbslq_s16(vmovl_u8(vceq_u8(codes, vdup_n_u8(3))), activation, contribution);
                contribution = vbslq_s16(vmovl_u8(vceq_u8(codes, vdup_n_u8(4))), triple, contribution);
                sum_low = vaddw_s16(sum_low, vget_low_s16(contribution));
                sum_high = vaddw_s16(sum_high, vget_high_s16(contribution));
                feature_codes += out_features;
            }
            alignas(16) int32_t partial_low[4];
            alignas(16) int32_t partial_high[4];
            vst1q_s32(partial_low, sum_low);
            vst1q_s32(partial_high, sum_high);
            for (int lane = 0; lane < 4; ++lane) {
                output_row[output_base + lane] = static_cast<int64_t>(partial_low[lane]);
                output_row[output_base + 4 + lane] = static_cast<int64_t>(partial_high[lane]);
            }
        }
#endif
        // Scalar implementation and tail. It is also the portable reference path.
        for (; output_base < out_features; ++output_base) {
            int64_t accumulator = 0;
            const uint8_t* feature_codes = weights_by_input + output_base;
            for (int feature = 0; feature < in_features; ++feature) {
                const uint8_t code = feature_codes[0];
                const uint8_t encoded = static_cast<uint8_t>(
                    static_cast<int>(input_row[feature]) + 128
                );
                accumulator += kContributionLookup[code][encoded];
                feature_codes += out_features;
            }
            output_row[output_base] = accumulator;
        }
        input_row += in_features;
        output_row += out_features;
    }
    return 0;
}

QMINI_ADDSUB_EXPORT int qmini_addsub_kernel_has_neon() {
    return QMINI_ADDSUB_HAS_NEON;
}

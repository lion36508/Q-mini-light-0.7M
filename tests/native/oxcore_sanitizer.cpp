#include "oxcore.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

bool check(bool condition, const char *label) {
    if (!condition) {
        std::fprintf(stderr, "OxCore sanitizer test failed: %s\n", label);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    bool seen[256] = {};
    size_t encoded = 0u;
    for (int q0 = -2; q0 <= 2; ++q0) {
        for (int q1 = -2; q1 <= 2; ++q1) {
            for (int q2 = -2; q2 <= 2; ++q2) {
                for (uint8_t octave = 0u; octave <= 1u; ++octave) {
                    uint8_t code = 0u;
                    if (!check(
                            ox_mirror63_pack_triplet(
                                static_cast<int8_t>(q0),
                                static_cast<int8_t>(q1),
                                static_cast<int8_t>(q2),
                                octave,
                                &code
                            ) == OX_OK,
                            "Mirror-63 pack")) {
                        return 1;
                    }
                    if (!check(!seen[code], "Mirror-63 encoding is unique")) {
                        return 1;
                    }
                    seen[code] = true;
                    ++encoded;

                    int8_t restored[3] = {};
                    uint8_t restored_octave = 0u;
                    if (!check(
                            ox_mirror63_unpack_triplet(
                                code, restored, &restored_octave
                            ) == OX_OK,
                            "Mirror-63 unpack")) {
                        return 1;
                    }
                    if (!check(
                            restored[0] == q0 &&
                            restored[1] == q1 &&
                            restored[2] == q2 &&
                            restored_octave == octave,
                            "Mirror-63 round trip")) {
                        return 1;
                    }
                }
            }
        }
    }
    if (!check(encoded == 250u, "Mirror-63 valid code count")) {
        return 1;
    }
    const uint8_t escapes[] = {0x3f, 0x40, 0x7f, 0xbf, 0xc0, 0xff};
    for (uint8_t code : escapes) {
        int8_t restored[3] = {};
        uint8_t octave = 0u;
        const ox_status status =
            ox_mirror63_unpack_triplet(code, restored, &octave);
        const ox_status expected = code == 0x3f ? OX_SKIP : OX_ERESERVED;
        if (!check(status == expected, "Mirror-63 escape handling")) {
            return 1;
        }
    }

    alignas(64) uint8_t arena_storage[1024] = {};
    ox_arena arena = {};
    if (!check(
            ox_arena_init(&arena, arena_storage, sizeof(arena_storage)) == OX_OK,
            "arena initialization")) {
        return 1;
    }
    void *allocation = nullptr;
    if (!check(
            ox_arena_alloc(&arena, 17u, 64u, &allocation) == OX_OK &&
            reinterpret_cast<uintptr_t>(allocation) % 64u == 0u,
            "arena alignment")) {
        return 1;
    }
    const size_t mark = ox_arena_mark(&arena);
    if (!check(
            ox_arena_alloc(&arena, 31u, 128u, &allocation) == OX_OK &&
            reinterpret_cast<uintptr_t>(allocation) % 128u == 0u,
            "arena second alignment")) {
        return 1;
    }
    if (!check(ox_arena_rewind(&arena, mark) == OX_OK, "arena rewind")) {
        return 1;
    }

    uint8_t weight_matrix[4] = {};
    if (!check(
            ox_mirror63_pack_triplet(1, -1, 0, 0u, &weight_matrix[0]) == OX_OK &&
            ox_mirror63_pack_triplet(2, 2, -2, 1u, &weight_matrix[1]) == OX_OK &&
            ox_mirror63_pack_triplet(-2, 2, 1, 0u, &weight_matrix[2]) == OX_OK &&
            ox_mirror63_pack_triplet(0, 0, 0, 0u, &weight_matrix[3]) == OX_OK,
            "linear weights packing")) {
        return 1;
    }
    const int8_t linear_input[] = {1, 2, 3, 4, -1, 5, -2, 3};
    int64_t linear_output[4] = {};
    if (!check(
            ox_linear_mirror63_acc_i8(
                linear_input, weight_matrix, 2u, 4u, 2u, linear_output
            ) == OX_OK,
            "integer linear projection")) {
        return 1;
    }
    if (!check(
            linear_output[0] == 15 && linear_output[1] == 5 &&
            linear_output[2] == 6 && linear_output[3] == 10,
            "integer linear projection values")) {
        return 1;
    }

    uint8_t conv_weights[2] = {};
    if (!check(
            ox_mirror63_pack_triplet(1, -1, 2, 1u, &conv_weights[0]) == OX_OK &&
            ox_mirror63_pack_triplet(2, 0, -1, 0u, &conv_weights[1]) == OX_OK,
            "causal convolution weights packing")) {
        return 1;
    }
    const int8_t conv_input[] = {1, 10, 2, 20, 3, 30, 4, 40};
    int64_t conv_output[8] = {};
    if (!check(
            ox_causal_depthwise_conv_mirror63_acc_i8(
                conv_input, 4u, 2u, conv_weights, 3u, conv_output
            ) == OX_OK,
            "causal depthwise convolution")) {
        return 1;
    }
    const int64_t expected_conv[] = {4, -10, 6, -20, 10, -10, 14, 0};
    for (size_t i = 0u; i < 8u; ++i) {
        if (!check(conv_output[i] == expected_conv[i], "causal convolution values")) {
            return 1;
        }
    }

    const int32_t initial_state[] = {100, -100};
    const int32_t injection[] = {10, 5, -5, 10, 20, -10};
    const uint16_t decay[] = {16384, 32768, 32768, 16384, 0, 32768};
    int32_t states[6] = {};
    int32_t final_state[2] = {};
    if (!check(
            ox_ssm_scan_q15_i32(
                initial_state, injection, decay, 3u, 1u, 2u, states, final_state
            ) == OX_OK,
            "Q15 SSM scan")) {
        return 1;
    }
    const int32_t expected_states[] = {60, -95, 55, -38, 20, -48};
    for (size_t i = 0u; i < 6u; ++i) {
        if (!check(states[i] == expected_states[i], "Q15 SSM state values")) {
            return 1;
        }
    }
    if (!check(
            final_state[0] == 20 && final_state[1] == -48,
            "Q15 SSM final cache")) {
        return 1;
    }

    const int32_t minimum_state[] = {
        std::numeric_limits<int32_t>::min()
    };
    const int32_t zero_injection[] = {0};
    const uint16_t full_decay[] = {32768u};
    int32_t minimum_output[] = {17};
    int32_t minimum_final[] = {19};
    if (!check(
            ox_ssm_scan_q15_i32(
                minimum_state, zero_injection, full_decay,
                1u, 1u, 1u, minimum_output, minimum_final
            ) == OX_OK &&
            minimum_output[0] == std::numeric_limits<int32_t>::min() &&
            minimum_final[0] == std::numeric_limits<int32_t>::min(),
            "Q15 SSM preserves INT32_MIN")) {
        return 1;
    }

    const int32_t cancel_initial[] = {0};
    const int32_t cancel_injection[] = {
        std::numeric_limits<int32_t>::max(),
        -std::numeric_limits<int32_t>::max()
    };
    const uint16_t cancel_decay[] = {0u, 32768u};
    int32_t cancel_states[2] = {};
    int32_t cancel_final[1] = {};
    if (!check(
            ox_ssm_scan_q15_i32(
                cancel_initial, cancel_injection, cancel_decay,
                2u, 1u, 1u, cancel_states, cancel_final
            ) == OX_OK &&
            cancel_states[0] == std::numeric_limits<int32_t>::max() &&
            cancel_states[1] == 0 && cancel_final[0] == 0,
            "Q15 SSM accepts safe decay/cancellation sequence")) {
        return 1;
    }

    const int32_t overflow_initial[] = {0};
    const int32_t overflow_injection[] = {
        std::numeric_limits<int32_t>::max(), 1
    };
    const uint16_t overflow_decay[] = {32768u, 32768u};
    int32_t untouched_states[] = {123, 456};
    int32_t untouched_final[] = {789};
    if (!check(
            ox_ssm_scan_q15_i32(
                overflow_initial, overflow_injection, overflow_decay,
                2u, 1u, 1u, untouched_states, untouched_final
            ) == OX_EOVERFLOW &&
            untouched_states[0] == 123 && untouched_states[1] == 456 &&
            untouched_final[0] == 789,
            "Q15 SSM overflow preflight is transactional")) {
        return 1;
    }

    alignas(8) uint8_t alias_storage[16] = {};
    const uint8_t one_weight[] = {0u};
    if (!check(
            ox_linear_mirror63_acc_i8(
                reinterpret_cast<const int8_t *>(alias_storage),
                one_weight, 1u, 1u, 1u,
                reinterpret_cast<int64_t *>(alias_storage)
            ) == OX_EALIAS,
            "linear projection rejects overlapping output")) {
        return 1;
    }

    int32_t size_dummy = 0;
    const size_t oversized_states =
        std::numeric_limits<size_t>::max() / sizeof(int32_t) + 1u;
    if (!check(
            ox_ssm_scan_q15_i32(
                &size_dummy, &size_dummy,
                reinterpret_cast<const uint16_t *>(&size_dummy),
                oversized_states, 1u, 1u, &size_dummy, &size_dummy
            ) == OX_EOVERFLOW,
            "SSM rejects byte-size overflow before dereference")) {
        return 1;
    }

    int8_t tiny_input[] = {0};
    int64_t tiny_output = 0;
    const size_t oversized_outputs =
        std::numeric_limits<size_t>::max() / sizeof(int64_t) + 1u;
    if (!check(
            ox_linear_mirror63_acc_i8(
                tiny_input, one_weight, 1u, 1u, oversized_outputs,
                &tiny_output
            ) == OX_EOVERFLOW,
            "linear rejects output byte-size overflow")) {
        return 1;
    }

    const size_t oversized_fwht =
        std::numeric_limits<size_t>::max() / sizeof(int64_t) + 1u;
    if (!check(
            ox_fwht_i64(&tiny_output, oversized_fwht) == OX_EOVERFLOW,
            "FWHT rejects byte-size overflow before dereference")) {
        return 1;
    }

    int16_t activation_table[256] = {};
    for (size_t index = 0u; index < 256u; ++index) {
        activation_table[index] = static_cast<int16_t>(
            static_cast<int>(index) - 128
        );
    }
    const uint8_t activation_indices[] = {0u, 128u, 255u};
    int16_t activation_values[3] = {};
    if (!check(
            ox_lut_u8_i16(
                activation_indices, activation_table, 3u, activation_values
            ) == OX_OK,
            "activation LUT")) {
        return 1;
    }
    if (!check(
            activation_values[0] == -128 &&
            activation_values[1] == 0 &&
            activation_values[2] == 127,
            "activation LUT values")) {
        return 1;
    }

    const int16_t router_scores[] = {2, 5, 5, -1, 7, 7, 1, 7};
    uint16_t selected_experts[2] = {};
    int16_t selected_scores[2] = {};
    if (!check(
            ox_top1_i16(
                router_scores, 2u, 4u, selected_experts, selected_scores
            ) == OX_OK,
            "Top-1 router")) {
        return 1;
    }
    if (!check(
            selected_experts[0] == 1u && selected_scores[0] == 5 &&
            selected_experts[1] == 0u && selected_scores[1] == 7,
            "stable Top-1 router values")) {
        return 1;
    }

    int64_t fwht[] = {3, -1, 4, 2};
    if (!check(ox_fwht_i64(fwht, 4u) == OX_OK, "integer FWHT")) {
        return 1;
    }
    if (!check(
            fwht[0] == 8 && fwht[1] == 6 && fwht[2] == -4 && fwht[3] == 2,
            "integer FWHT values")) {
        return 1;
    }

    double logits[16] = {};
    double matrix[16] = {};
    if (!check(ox_sinkhorn4(logits, matrix, 20u) == OX_OK, "Sinkhorn-4")) {
        return 1;
    }
    for (size_t row = 0u; row < 4u; ++row) {
        double sum = 0.0;
        for (size_t column = 0u; column < 4u; ++column) {
            sum += matrix[row * 4u + column];
        }
        if (!check(std::isfinite(sum) && std::fabs(sum - 1.0) < 1e-9,
                   "Sinkhorn row sums")) {
            return 1;
        }
    }

    std::puts("OxCore ASan/UBSan native tests passed.");
    return 0;
}

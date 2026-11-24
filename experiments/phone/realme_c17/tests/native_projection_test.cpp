#include <cstdint>
#include <cstdio>

extern "C" int qmini_addsub_projection_i8(const int8_t*, const uint8_t*, int64_t*, int, int, int);

static int64_t reference_term(int activation, uint8_t code) {
    const int triple = activation + activation + activation;
    switch (code) {
        case 0: return -triple;
        case 1: return -activation;
        case 2: return 0;
        case 3: return activation;
        case 4: return triple;
        default: return 0;
    }
}

int main() {
    for (int activation = -127; activation <= 127; ++activation) {
        int8_t input[1] = {static_cast<int8_t>(activation)};
        for (uint8_t code = 0; code <= 4; ++code) {
            uint8_t weight[1] = {code};
            int64_t actual[1] = {0};
            if (qmini_addsub_projection_i8(input, weight, actual, 1, 1, 1) != 0 ||
                actual[0] != reference_term(activation, code)) {
                std::fprintf(stderr, "single-pair mismatch activation=%d code=%u\n", activation, static_cast<unsigned int>(code));
                return 1;
            }
        }
    }
    constexpr int rows = 2, features = 7, outputs = 11;
    int8_t inputs[rows][features] = {{3, -7, 11, -19, 23, -31, 41}, {-5, 13, -17, 29, -37, 43, -53}};
    uint8_t weights[features][outputs] = {};
    for (int f = 0; f < features; ++f) for (int o = 0; o < outputs; ++o) weights[f][o] = static_cast<uint8_t>((f + o) % 5);
    int64_t actual[rows][outputs] = {};
    if (qmini_addsub_projection_i8(&inputs[0][0], &weights[0][0], &actual[0][0], rows, features, outputs) != 0) return 2;
    for (int r = 0; r < rows; ++r) for (int o = 0; o < outputs; ++o) {
        int64_t expected = 0;
        for (int f = 0; f < features; ++f) expected += reference_term(inputs[r][f], weights[f][o]);
        if (actual[r][o] != expected) { std::fprintf(stderr, "batch mismatch row=%d output=%d\n", r, o); return 3; }
    }
    int8_t invalid_activation[1] = {static_cast<int8_t>(-128)};
    uint8_t valid_weight[1] = {2};
    int64_t untouched[1] = {123456};
    if (qmini_addsub_projection_i8(invalid_activation, valid_weight, untouched, 1, 1, 1) != 3 || untouched[0] != 123456) return 4;
    int8_t valid_activation[1] = {1};
    uint8_t invalid_weight[1] = {5};
    if (qmini_addsub_projection_i8(valid_activation, invalid_weight, untouched, 1, 1, 1) != 4 || untouched[0] != 123456) return 5;
    return 0;
}

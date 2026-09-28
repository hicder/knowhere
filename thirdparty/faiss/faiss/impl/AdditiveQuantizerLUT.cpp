// -*- c++ -*-

#include <faiss/impl/AdditiveQuantizerLUT.h>

#include <atomic>

#include <faiss/FaissHook.h>
#include <faiss/impl/AdditiveQuantizer.h>

namespace faiss {

namespace {

std::atomic<AQLutKernel> aq_lut_kernel{AQLutKernel::AUTO};

bool all_8bit(const AdditiveQuantizer& aq) {
    for (size_t m = 0; m < aq.M; m++) {
        if (aq.nbits[m] != 8) {
            return false;
        }
    }
    return aq.M > 0;
}

} // namespace

void set_aq_lut_kernel(AQLutKernel kernel) {
    aq_lut_kernel.store(kernel, std::memory_order_relaxed);
}

AQLutKernel get_aq_lut_kernel() {
    return aq_lut_kernel.load(std::memory_order_relaxed);
}

AQLutKernel resolve_aq_lut_kernel(const AdditiveQuantizer& aq) {
    AQLutKernel kernel = get_aq_lut_kernel();
    if (kernel == AQLutKernel::GENERIC || !all_8bit(aq)) {
        return AQLutKernel::GENERIC;
    }
#if defined(__x86_64__)
    bool has_avx512 = use_avx512 && cpu_support_avx512();
#else
    bool has_avx512 = false;
#endif
    if (kernel == AQLutKernel::AUTO) {
        return AQLutKernel::SCALAR_8BIT;
    }
    if (kernel == AQLutKernel::AVX512_8BIT && !has_avx512) {
        return AQLutKernel::SCALAR_8BIT;
    }
    return kernel;
}

const char* aq_lut_kernel_name(AQLutKernel kernel) {
    switch (kernel) {
        case AQLutKernel::AUTO:
            return "auto";
        case AQLutKernel::GENERIC:
            return "generic";
        case AQLutKernel::SCALAR_8BIT:
            return "scalar_8bit";
        case AQLutKernel::AVX512_8BIT:
            return "avx512_8bit";
    }
    return "unknown";
}

float aq_lut_accumulate_8bit(const uint8_t* code, const float* LUT, size_t M) {
    float accu = 0;
    for (size_t m = 0; m < M; m++) {
        accu += LUT[m * 256 + code[m]];
    }
    return accu;
}

void aq_lut_accumulate_8bit_batch4(
        const uint8_t* c0,
        const uint8_t* c1,
        const uint8_t* c2,
        const uint8_t* c3,
        const float* LUT,
        size_t M,
        float* out) {
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (size_t m = 0; m < M; m++) {
        const float* table = LUT + m * 256;
        a0 += table[c0[m]];
        a1 += table[c1[m]];
        a2 += table[c2[m]];
        a3 += table[c3[m]];
    }
    out[0] = a0;
    out[1] = a1;
    out[2] = a2;
    out[3] = a3;
}

} // namespace faiss

// -*- c++ -*-

#pragma once

#include <cstddef>
#include <cstdint>

namespace faiss {

struct AdditiveQuantizer;

/** Kernel used to accumulate LUT inner products when computing distances to
 * additive quantizer codes (AQDistanceComputerLUT, eg. HNSW_PRQ search).
 *
 * The 8-bit kernels only apply when every codebook uses 8 bits; otherwise
 * GENERIC is used regardless of the setting.
 */
enum class AQLutKernel : int {
    /// SCALAR_8BIT: the batch of 4 path used by HNSW search is gather
    /// throughput bound with AVX512_8BIT, and not faster on Zen 4
    AUTO = 0,
    /// original bit-reader based implementation
    GENERIC = 1,
    /// direct byte indexing, fused batch of 4. Bitwise identical to GENERIC.
    SCALAR_8BIT = 2,
    /// AVX-512 gathers over 16 codebooks, falls back to SCALAR_8BIT when
    /// unsupported or use_avx512 is off. Sums are reassociated, so results
    /// differ from GENERIC by float rounding.
    AVX512_8BIT = 3,
};

void set_aq_lut_kernel(AQLutKernel kernel);
AQLutKernel get_aq_lut_kernel();

/// the kernel actually used for aq given the current setting and CPU
AQLutKernel resolve_aq_lut_kernel(const AdditiveQuantizer& aq);

const char* aq_lut_kernel_name(AQLutKernel kernel);

/// sum_m LUT[m * 256 + code[m]], in order of m
float aq_lut_accumulate_8bit(const uint8_t* code, const float* LUT, size_t M);

/// out[j] = aq_lut_accumulate_8bit(c_j, LUT, M) for 4 codes at once
void aq_lut_accumulate_8bit_batch4(
        const uint8_t* c0,
        const uint8_t* c1,
        const uint8_t* c2,
        const uint8_t* c3,
        const float* LUT,
        size_t M,
        float* out);

#if defined(__x86_64__)
float aq_lut_accumulate_8bit_avx512(
        const uint8_t* code,
        const float* LUT,
        size_t M);

void aq_lut_accumulate_8bit_batch4_avx512(
        const uint8_t* c0,
        const uint8_t* c1,
        const uint8_t* c2,
        const uint8_t* c3,
        const float* LUT,
        size_t M,
        float* out);
#endif

} // namespace faiss

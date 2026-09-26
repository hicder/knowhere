// -*- c++ -*-

#include <faiss/impl/AdditiveQuantizerLUT.h>

#include <immintrin.h>

namespace faiss {

namespace {

// Codebook m of a uniform 8-bit LUT starts at m * 256. A block covers 16
// consecutive codebooks, lane l indexes codebook (m + l).
inline __m512i block_offsets() {
    return _mm512_setr_epi32(
            0 * 256,
            1 * 256,
            2 * 256,
            3 * 256,
            4 * 256,
            5 * 256,
            6 * 256,
            7 * 256,
            8 * 256,
            9 * 256,
            10 * 256,
            11 * 256,
            12 * 256,
            13 * 256,
            14 * 256,
            15 * 256);
}

inline __m512 gather_block(
        const uint8_t* code,
        const float* table,
        __m512i offsets) {
    __m512i idx = _mm512_cvtepu8_epi32(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(code)));
    return _mm512_i32gather_ps(_mm512_add_epi32(idx, offsets), table, 4);
}

inline __m512 gather_tail(
        const uint8_t* code,
        const float* table,
        __m512i offsets,
        __mmask16 mask) {
    __m512i idx = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(mask, code));
    return _mm512_mask_i32gather_ps(
            _mm512_setzero_ps(),
            mask,
            _mm512_add_epi32(idx, offsets),
            table,
            4);
}

} // namespace

// Even blocks go to acc0 and odd blocks to acc1, in both kernels, so that a
// batched result is bitwise equal to the corresponding single result.

float aq_lut_accumulate_8bit_avx512(
        const uint8_t* code,
        const float* LUT,
        size_t M) {
    const __m512i offsets = block_offsets();
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    size_t m = 0;
    for (; m + 32 <= M; m += 32) {
        acc0 = _mm512_add_ps(
                acc0, gather_block(code + m, LUT + m * 256, offsets));
        acc1 = _mm512_add_ps(
                acc1,
                gather_block(code + m + 16, LUT + (m + 16) * 256, offsets));
    }
    if (m + 16 <= M) {
        acc0 = _mm512_add_ps(
                acc0, gather_block(code + m, LUT + m * 256, offsets));
        m += 16;
        if (m < M) {
            __mmask16 mask = (__mmask16)((1u << (M - m)) - 1);
            acc1 = _mm512_add_ps(
                    acc1, gather_tail(code + m, LUT + m * 256, offsets, mask));
        }
    } else if (m < M) {
        __mmask16 mask = (__mmask16)((1u << (M - m)) - 1);
        acc0 = _mm512_add_ps(
                acc0, gather_tail(code + m, LUT + m * 256, offsets, mask));
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

void aq_lut_accumulate_8bit_batch4_avx512(
        const uint8_t* c0,
        const uint8_t* c1,
        const uint8_t* c2,
        const uint8_t* c3,
        const float* LUT,
        size_t M,
        float* out) {
    const __m512i offsets = block_offsets();
    __m512 a00 = _mm512_setzero_ps(), a01 = _mm512_setzero_ps();
    __m512 a10 = _mm512_setzero_ps(), a11 = _mm512_setzero_ps();
    __m512 a20 = _mm512_setzero_ps(), a21 = _mm512_setzero_ps();
    __m512 a30 = _mm512_setzero_ps(), a31 = _mm512_setzero_ps();
    size_t m = 0;
    for (; m + 32 <= M; m += 32) {
        const float* t0 = LUT + m * 256;
        const float* t1 = LUT + (m + 16) * 256;
        a00 = _mm512_add_ps(a00, gather_block(c0 + m, t0, offsets));
        a10 = _mm512_add_ps(a10, gather_block(c1 + m, t0, offsets));
        a20 = _mm512_add_ps(a20, gather_block(c2 + m, t0, offsets));
        a30 = _mm512_add_ps(a30, gather_block(c3 + m, t0, offsets));
        a01 = _mm512_add_ps(a01, gather_block(c0 + m + 16, t1, offsets));
        a11 = _mm512_add_ps(a11, gather_block(c1 + m + 16, t1, offsets));
        a21 = _mm512_add_ps(a21, gather_block(c2 + m + 16, t1, offsets));
        a31 = _mm512_add_ps(a31, gather_block(c3 + m + 16, t1, offsets));
    }
    if (m + 16 <= M) {
        const float* t0 = LUT + m * 256;
        a00 = _mm512_add_ps(a00, gather_block(c0 + m, t0, offsets));
        a10 = _mm512_add_ps(a10, gather_block(c1 + m, t0, offsets));
        a20 = _mm512_add_ps(a20, gather_block(c2 + m, t0, offsets));
        a30 = _mm512_add_ps(a30, gather_block(c3 + m, t0, offsets));
        m += 16;
        if (m < M) {
            const float* t1 = LUT + m * 256;
            __mmask16 mask = (__mmask16)((1u << (M - m)) - 1);
            a01 = _mm512_add_ps(a01, gather_tail(c0 + m, t1, offsets, mask));
            a11 = _mm512_add_ps(a11, gather_tail(c1 + m, t1, offsets, mask));
            a21 = _mm512_add_ps(a21, gather_tail(c2 + m, t1, offsets, mask));
            a31 = _mm512_add_ps(a31, gather_tail(c3 + m, t1, offsets, mask));
        }
    } else if (m < M) {
        const float* t0 = LUT + m * 256;
        __mmask16 mask = (__mmask16)((1u << (M - m)) - 1);
        a00 = _mm512_add_ps(a00, gather_tail(c0 + m, t0, offsets, mask));
        a10 = _mm512_add_ps(a10, gather_tail(c1 + m, t0, offsets, mask));
        a20 = _mm512_add_ps(a20, gather_tail(c2 + m, t0, offsets, mask));
        a30 = _mm512_add_ps(a30, gather_tail(c3 + m, t0, offsets, mask));
    }
    out[0] = _mm512_reduce_add_ps(_mm512_add_ps(a00, a01));
    out[1] = _mm512_reduce_add_ps(_mm512_add_ps(a10, a11));
    out[2] = _mm512_reduce_add_ps(_mm512_add_ps(a20, a21));
    out[3] = _mm512_reduce_add_ps(_mm512_add_ps(a30, a31));
}

} // namespace faiss

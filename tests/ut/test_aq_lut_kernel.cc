// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "faiss/IndexAdditiveQuantizer.h"
#include "faiss/impl/AdditiveQuantizerLUT.h"
#include "faiss/utils/distances.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "simd/hook.h"
#include "utils.h"

namespace {

using ST = faiss::AdditiveQuantizer::Search_type_t;

// Residual quantizer index with random codebooks and codes, skipping training so that any M is cheap to set up.
// Row 0 uses codeword 0 everywhere and row 1 uses the largest codeword everywhere.
std::unique_ptr<faiss::IndexResidualQuantizer>
MakeRandomAQIndex(int d, const std::vector<size_t>& nbits, faiss::MetricType metric, ST st, size_t n, uint32_t seed) {
    auto index = std::make_unique<faiss::IndexResidualQuantizer>(d, nbits, metric, st);
    auto& rq = index->rq;
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;

    rq.codebooks.resize(rq.total_codebook_size * d);
    for (auto& v : rq.codebooks) {
        v = normal(rng);
    }
    rq.is_trained = true;
    index->is_trained = true;

    const size_t M = rq.M;
    std::vector<int32_t> codes(n * M);
    for (size_t i = 0; i < n; i++) {
        for (size_t m = 0; m < M; m++) {
            const int32_t ksub = 1 << nbits[m];
            int32_t c = std::uniform_int_distribution<int32_t>(0, ksub - 1)(rng);
            if (i == 0) {
                c = 0;
            } else if (i == 1) {
                c = ksub - 1;
            }
            codes[i * M + m] = c;
        }
    }

    std::vector<float> x(n * d);
    rq.decode_unpacked(codes.data(), x.data(), n);
    std::vector<float> norms(n);
    faiss::fvec_norms_L2sqr(norms.data(), x.data(), d, n);
    // exact 1D k-means needs at least 256 distinct values, which small M cannot provide
    std::vector<float> norms_train(1024);
    for (size_t i = 0; i < norms_train.size(); i++) {
        norms_train[i] = norms[i % n] * (1 + 1e-3f * normal(rng));
    }
    rq.train_norm(norms_train.size(), norms_train.data());

    index->codes.resize(n * index->code_size);
    rq.pack_codes(n, codes.data(), index->codes.data(), -1, norms.data());
    index->ntotal = n;
    return index;
}

std::vector<float>
RandomQuery(int d, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;
    std::vector<float> q(d);
    for (auto& v : q) {
        v = normal(rng);
    }
    return q;
}

struct KernelGuard {
    faiss::AQLutKernel saved = faiss::get_aq_lut_kernel();
    ~KernelGuard() {
        faiss::set_aq_lut_kernel(saved);
    }
};

struct KernelOutputs {
    std::vector<float> single;
    std::vector<float> batch;
};

// single[i] = distance to id i, batch = distances for all id quads, flattened
KernelOutputs
ComputeWith(faiss::AQLutKernel kernel, const faiss::IndexResidualQuantizer& index, const float* query,
            const std::vector<std::array<faiss::idx_t, 4>>& quads, faiss::AQLutKernel* resolved) {
    faiss::set_aq_lut_kernel(kernel);
    *resolved = faiss::resolve_aq_lut_kernel(*index.aq);
    std::unique_ptr<faiss::FlatCodesDistanceComputer> dc(index.get_FlatCodesDistanceComputer());
    dc->set_query(query);
    KernelOutputs out;
    for (faiss::idx_t i = 0; i < index.ntotal; i++) {
        out.single.push_back((*dc)(i));
    }
    for (const auto& q : quads) {
        float d0, d1, d2, d3;
        dc->distances_batch_4(q[0], q[1], q[2], q[3], d0, d1, d2, d3);
        out.batch.insert(out.batch.end(), {d0, d1, d2, d3});
    }
    return out;
}

bool
BitwiseEqual(float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

}  // namespace

TEST_CASE("Test AQ LUT distance kernels", "[aq lut kernel]") {
    KernelGuard guard;
    const int d = 8;
    const size_t n = 300;

    auto [metric, st] = GENERATE(table<faiss::MetricType, ST>({
        {faiss::METRIC_INNER_PRODUCT, ST::ST_LUT_nonorm},
        {faiss::METRIC_L2, ST::ST_LUT_nonorm},
        {faiss::METRIC_L2, ST::ST_norm_float},
        {faiss::METRIC_L2, ST::ST_norm_qint8},
        {faiss::METRIC_L2, ST::ST_norm_qint4},
        {faiss::METRIC_L2, ST::ST_norm_cqint8},
        {faiss::METRIC_L2, ST::ST_norm_cqint4},
    }));
    const size_t M = GENERATE(as<size_t>{}, 1, 3, 4, 5, 15, 16, 17, 31, 32, 33, 48, 384);
    CAPTURE(metric, st, M);

    auto index = MakeRandomAQIndex(d, std::vector<size_t>(M, 8), metric, st, n, 1234 + M);
    const auto query = RandomQuery(d, 99);

    std::mt19937 rng(7);
    std::uniform_int_distribution<faiss::idx_t> id_dist(0, n - 1);
    const auto last = faiss::idx_t(n - 1);
    std::vector<std::array<faiss::idx_t, 4>> quads = {{0, 1, 2, 3}, {1, 0, 1, 0}, {5, 5, 5, 5}, {last, 0, last - 1, 1}};
    for (int i = 0; i < 64; i++) {
        quads.push_back({id_dist(rng), id_dist(rng), id_dist(rng), id_dist(rng)});
    }

    faiss::AQLutKernel resolved;
    const auto ref = ComputeWith(faiss::AQLutKernel::GENERIC, *index, query.data(), quads, &resolved);
    REQUIRE(resolved == faiss::AQLutKernel::GENERIC);
    for (size_t j = 0; j < quads.size() * 4; j++) {
        REQUIRE(BitwiseEqual(ref.batch[j], ref.single[quads[j / 4][j % 4]]));
    }

    SECTION("scalar kernel is bitwise identical to generic") {
        const auto res = ComputeWith(faiss::AQLutKernel::SCALAR_8BIT, *index, query.data(), quads, &resolved);
        REQUIRE(resolved == faiss::AQLutKernel::SCALAR_8BIT);
        for (size_t i = 0; i < n; i++) {
            CAPTURE(i);
            REQUIRE(BitwiseEqual(res.single[i], ref.single[i]));
        }
        for (size_t j = 0; j < res.batch.size(); j++) {
            CAPTURE(j);
            REQUIRE(BitwiseEqual(res.batch[j], ref.batch[j]));
        }
    }

#if defined(__x86_64__)
    SECTION("avx512 kernel matches generic up to rounding") {
        if (!faiss::cpu_support_avx512()) {
            SKIP("AVX-512 is not supported on this CPU");
        }
        const auto res = ComputeWith(faiss::AQLutKernel::AVX512_8BIT, *index, query.data(), quads, &resolved);
        REQUIRE(resolved == faiss::AQLutKernel::AVX512_8BIT);
        // LUT entries are O(sqrt(d)), accumulated over M codebooks, and scaled by 2 for L2
        const float tol = 1e-5f * std::sqrt(float(d)) * M * 2;
        for (size_t i = 0; i < n; i++) {
            CAPTURE(i);
            REQUIRE(std::abs(res.single[i] - ref.single[i]) <= tol);
        }
        for (size_t j = 0; j < res.batch.size(); j++) {
            CAPTURE(j);
            REQUIRE(BitwiseEqual(res.batch[j], res.single[quads[j / 4][j % 4]]));
        }
    }
#endif
}

TEST_CASE("Test AQ LUT kernel falls back for non 8-bit codes", "[aq lut kernel]") {
    KernelGuard guard;
    const int d = 8;
    const size_t n = 300;
    auto index = MakeRandomAQIndex(d, {8, 6, 8, 4, 8}, faiss::METRIC_L2, ST::ST_norm_float, n, 42);
    const auto query = RandomQuery(d, 5);
    std::vector<std::array<faiss::idx_t, 4>> quads = {{0, 1, 2, 3}, {7, 7, 100, 299}};

    faiss::AQLutKernel resolved;
    const auto ref = ComputeWith(faiss::AQLutKernel::GENERIC, *index, query.data(), quads, &resolved);
    for (auto kernel : {faiss::AQLutKernel::AUTO, faiss::AQLutKernel::SCALAR_8BIT, faiss::AQLutKernel::AVX512_8BIT}) {
        const auto res = ComputeWith(kernel, *index, query.data(), quads, &resolved);
        REQUIRE(resolved == faiss::AQLutKernel::GENERIC);
        REQUIRE(res.single == ref.single);
        REQUIRE(res.batch == ref.batch);
    }
}

TEST_CASE("Test AQ LUT kernel knowhere config", "[aq lut kernel]") {
    KernelGuard guard;
    using K = knowhere::KnowhereConfig;
    for (auto kernel : {K::AQ_GENERIC, K::AQ_SCALAR, K::AQ_AVX512, K::AQ_AUTO}) {
        K::SetAQDistanceKernel(kernel);
        REQUIRE(K::GetAQDistanceKernel() == kernel);
    }
}

TEST_CASE("Test HNSW_PRQ search with AQ LUT kernels", "[aq lut kernel]") {
    KernelGuard guard;
    using K = knowhere::KnowhereConfig;
    const int64_t nb = 2000, nq = 20, dim = 48, topk = 10;
    const auto metric =
        GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP, knowhere::metric::COSINE);
    CAPTURE(metric);

    const auto version = GenTestVersionList();
    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim, 43);

    knowhere::Json json;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::meta::METRIC_TYPE] = metric;
    json[knowhere::meta::TOPK] = topk;
    json[knowhere::indexparam::HNSW_M] = 16;
    json[knowhere::indexparam::EFCONSTRUCTION] = 96;
    json[knowhere::indexparam::EF] = 64;
    // 12 splits x 2 residual codebooks = 24 codebooks, not a multiple of the 16-wide AVX-512 block
    json[knowhere::indexparam::M] = 12;
    json[knowhere::indexparam::PRQ_NUM] = 2;
    json[knowhere::indexparam::NBITS] = 8;

    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_PRQ, version);
    REQUIRE(idx.has_value());
    auto index = idx.value();
    REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);

    auto search = [&](K::AQDistanceKernel kernel) {
        K::SetAQDistanceKernel(kernel);
        auto res = index.Search(query_ds, json, nullptr);
        REQUIRE(res.has_value());
        return res.value();
    };

    const auto generic = search(K::AQ_GENERIC);
    const auto scalar = search(K::AQ_SCALAR);
    const auto* g_ids = generic->GetIds();
    const auto* g_dis = generic->GetDistance();
    for (int64_t i = 0; i < nq * topk; i++) {
        CAPTURE(i);
        REQUIRE(scalar->GetIds()[i] == g_ids[i]);
        REQUIRE(BitwiseEqual(scalar->GetDistance()[i], g_dis[i]));
    }

    const auto avx512 = search(K::AQ_AVX512);
    REQUIRE(GetKNNRecall(*generic, *avx512) >= 0.95f);
    for (int64_t i = 0; i < nq * topk; i++) {
        if (avx512->GetIds()[i] == g_ids[i]) {
            REQUIRE(avx512->GetDistance()[i] == Catch::Approx(g_dis[i]).epsilon(1e-4).margin(1e-2));
        }
    }
}

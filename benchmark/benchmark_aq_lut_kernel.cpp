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

// Compares the AQ LUT distance kernels (generic / scalar_8bit / avx512_8bit) used by HNSW_PRQ search:
//   1. micro: distance computer throughput on a trained PRQ, random candidate ids, single and batch-of-4 calls
//   2. hnsw:  HNSW graph search (the knowhere v2_hnsw_searcher) over the same PRQ codes, with the per-query LUTs
//             built before timing, so QPS and latency only cover graph traversal and distance computations
//   3. e2e:   knowhere HNSW_PRQ search QPS, latency percentiles and recall, including LUT construction
// Data is synthetic (gaussian mixture), no HDF5 needed. PRQ training dominates the setup time (minutes for the
// default 192x2x8 layout); micro and hnsw share one trained PRQ, e2e trains its own.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "faiss/IndexAdditiveQuantizer.h"
#include "faiss/IndexHNSW.h"
#include "faiss/cppcontrib/knowhere/impl/HnswSearcher.h"
#include "faiss/cppcontrib/knowhere/utils/Bitset.h"
#include "faiss/impl/AdditiveQuantizerLUT.h"
#include "faiss/impl/IDSelector.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/version.h"
#include "simd/hook.h"

namespace {

using Clock = std::chrono::steady_clock;

double
SecondsSince(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

struct Options {
    int dim = 768;
    int prq_m = 192;
    int prq_nrq = 2;
    int hnsw_m = 50;
    int ef_construction = 100;
    int ef = 64;
    int topk = 10;
    int64_t nb = 20000;
    int64_t ntrain = 10000;
    int64_t nq = 1000;
    int64_t micro_candidates = 1 << 20;
    int micro_queries = 16;
    int repeats = 5;
    int search_threads = 0;
    std::vector<int> hnsw_threads;
    int hnsw_passes = 10;
    std::string metric = "L2";
    bool run_micro = true;
    bool run_hnsw = true;
    bool run_e2e = true;
};

std::vector<int>
ParseIntList(const std::string& s) {
    std::vector<int> out;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t end = s.find(',', pos);
        if (end == std::string::npos) {
            end = s.size();
        }
        out.push_back(std::atoi(s.substr(pos, end - pos).c_str()));
        pos = end + 1;
    }
    return out;
}

void
Usage(const char* prog) {
    std::printf(
        "usage: %s [options]\n"
        "  --dim N              vector dimension (768)\n"
        "  --m N                PRQ splits (192)\n"
        "  --nrq N              PRQ residual quantizers per split (2), nbits is always 8\n"
        "  --hnsw-m N           HNSW graph degree (50)\n"
        "  --efc N              HNSW efConstruction (100)\n"
        "  --ef N               HNSW search ef (64)\n"
        "  --topk N             (10)\n"
        "  --nb N               database size (20000)\n"
        "  --ntrain N           PRQ training vectors for the micro benchmark (10000)\n"
        "  --nq N               hnsw and e2e queries (1000). hnsw keeps one LUT per query in memory\n"
        "                       (total_codebook_size * 4 bytes each, 384 KiB for 192x2x8)\n"
        "  --hnsw-threads LIST  hnsw search thread counts (1,4,16,<hardware concurrency>)\n"
        "  --hnsw-passes N      hnsw passes over the queries per measurement (10)\n"
        "  --candidates N       micro benchmark candidates per query (1048576)\n"
        "  --micro-queries N    micro benchmark queries (16)\n"
        "  --repeats N          repetitions per measurement (5)\n"
        "  --search-threads N   knowhere search thread pool size (hardware concurrency)\n"
        "  --metric L2|IP|COSINE (L2)\n"
        "  --modes LIST         any of micro,hnsw,e2e (micro,hnsw,e2e)\n",
        prog);
}

bool
ParseArgs(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--dim") {
            o.dim = std::atoi(next());
        } else if (a == "--m") {
            o.prq_m = std::atoi(next());
        } else if (a == "--nrq") {
            o.prq_nrq = std::atoi(next());
        } else if (a == "--hnsw-m") {
            o.hnsw_m = std::atoi(next());
        } else if (a == "--efc") {
            o.ef_construction = std::atoi(next());
        } else if (a == "--ef") {
            o.ef = std::atoi(next());
        } else if (a == "--topk") {
            o.topk = std::atoi(next());
        } else if (a == "--nb") {
            o.nb = std::atoll(next());
        } else if (a == "--ntrain") {
            o.ntrain = std::atoll(next());
        } else if (a == "--nq") {
            o.nq = std::atoll(next());
        } else if (a == "--candidates") {
            o.micro_candidates = std::atoll(next());
        } else if (a == "--micro-queries") {
            o.micro_queries = std::atoi(next());
        } else if (a == "--repeats") {
            o.repeats = std::atoi(next());
        } else if (a == "--search-threads") {
            o.search_threads = std::atoi(next());
        } else if (a == "--metric") {
            o.metric = next();
        } else if (a == "--hnsw-threads") {
            o.hnsw_threads = ParseIntList(next());
        } else if (a == "--hnsw-passes") {
            o.hnsw_passes = std::atoi(next());
        } else if (a == "--modes") {
            const std::string modes = std::string(",") + next() + ",";
            o.run_micro = modes.find(",micro,") != std::string::npos;
            o.run_hnsw = modes.find(",hnsw,") != std::string::npos;
            o.run_e2e = modes.find(",e2e,") != std::string::npos;
        } else if (a == "--help" || a == "-h") {
            Usage(argv[0]);
            std::exit(0);
        } else {
            Usage(argv[0]);
            return false;
        }
    }
    if (o.dim % o.prq_m != 0) {
        std::fprintf(stderr, "dim (%d) must be divisible by m (%d)\n", o.dim, o.prq_m);
        return false;
    }
    if (o.metric != "L2" && o.metric != "IP" && o.metric != "COSINE") {
        std::fprintf(stderr, "unknown metric %s\n", o.metric.c_str());
        return false;
    }
    if (o.hnsw_threads.empty()) {
        const int hw = std::max(1u, std::thread::hardware_concurrency());
        for (int t : {1, 4, 16}) {
            if (t < hw) {
                o.hnsw_threads.push_back(t);
            }
        }
        o.hnsw_threads.push_back(hw);
    }
    o.micro_candidates -= o.micro_candidates % 4;
    return true;
}

// gaussian mixture, so that codes are not uniformly distributed like purely random data
std::vector<float>
GenData(int64_t n, int dim, uint32_t seed) {
    const int n_centers = 64;
    std::mt19937 rng(seed);
    std::mt19937 center_rng(12345);
    std::normal_distribution<float> normal;
    std::vector<float> centers(size_t(n_centers) * dim);
    for (auto& v : centers) {
        v = normal(center_rng) * 2.0f;
    }
    std::uniform_int_distribution<int> pick(0, n_centers - 1);
    std::vector<float> x(size_t(n) * dim);
    for (int64_t i = 0; i < n; i++) {
        const float* c = centers.data() + size_t(pick(rng)) * dim;
        for (int j = 0; j < dim; j++) {
            x[i * dim + j] = c[j] + normal(rng);
        }
    }
    return x;
}

knowhere::DataSetPtr
ToDataSet(const std::vector<float>& x, int64_t n, int dim) {
    auto* data = new float[size_t(n) * dim];
    std::memcpy(data, x.data(), sizeof(float) * n * dim);
    return knowhere::GenDataSet(n, dim, data);
}

struct Stats {
    double median = 0, min = 0, max = 0;
};

Stats
Summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return {v[v.size() / 2], v.front(), v.back()};
}

const std::vector<faiss::AQLutKernel> kKernels = {faiss::AQLutKernel::GENERIC, faiss::AQLutKernel::SCALAR_8BIT,
                                                  faiss::AQLutKernel::AVX512_8BIT};

std::vector<faiss::AQLutKernel>
AvailableKernels(const faiss::AdditiveQuantizer& aq) {
    std::vector<faiss::AQLutKernel> out;
    for (auto k : kKernels) {
        faiss::set_aq_lut_kernel(k);
        if (faiss::resolve_aq_lut_kernel(aq) == k) {
            out.push_back(k);
        } else {
            std::printf("  kernel %s unavailable here, skipped\n", faiss::aq_lut_kernel_name(k));
        }
    }
    return out;
}

void
NormalizeRows(std::vector<float>& x, int dim) {
    for (size_t i = 0; i < x.size(); i += dim) {
        float norm = 0;
        for (int j = 0; j < dim; j++) {
            norm += x[i + j] * x[i + j];
        }
        norm = std::sqrt(norm);
        for (int j = 0; j < dim; j++) {
            x[i + j] /= norm;
        }
    }
}

// COSINE is benchmarked as IP over normalized vectors, which is what knowhere does
std::vector<float>
GenMetricData(const Options& o, int64_t n, uint32_t seed) {
    auto x = GenData(n, o.dim, seed);
    if (o.metric == "COSINE") {
        NormalizeRows(x, o.dim);
    }
    return x;
}

/// trained and populated PRQ shared by the micro and hnsw benchmarks
struct PrqFixture {
    bool is_ip = false;
    std::vector<float> base;
    std::unique_ptr<faiss::IndexProductResidualQuantizer> index;
};

PrqFixture
BuildPrq(const Options& o) {
    PrqFixture f;
    f.is_ip = o.metric != "L2";
    const auto metric = f.is_ip ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
    const auto st = f.is_ip ? faiss::AdditiveQuantizer::ST_LUT_nonorm : faiss::AdditiveQuantizer::ST_norm_float;

    std::printf("\n=== PRQ m=%d nrq=%d nbits=8 (aq.M=%d), dim=%d, nb=%ld, ntrain=%ld, metric=%s ===\n", o.prq_m,
                o.prq_nrq, o.prq_m * o.prq_nrq, o.dim, (long)o.nb, (long)o.ntrain, o.metric.c_str());
    f.index = std::make_unique<faiss::IndexProductResidualQuantizer>(o.dim, o.prq_m, o.prq_nrq, 8, metric, st);
    auto t0 = Clock::now();
    auto train = GenMetricData(o, o.ntrain, 1);
    f.index->train(o.ntrain, train.data());
    std::printf("  train: %.1fs\n", SecondsSince(t0));
    t0 = Clock::now();
    f.base = GenMetricData(o, o.nb, 2);
    f.index->add(o.nb, f.base.data());
    std::printf("  add: %.1fs, code_size=%zu bytes, LUT=%zu KiB\n", SecondsSince(t0), f.index->code_size,
                f.index->aq->total_codebook_size * sizeof(float) / 1024);
    return f;
}

/*****************************************************************************
 * micro benchmark
 *****************************************************************************/

void
RunMicro(const Options& o, const PrqFixture& f) {
    const auto& index = *f.index;
    std::printf("\n=== micro: %d queries x %ld random candidates ===\n", o.micro_queries, (long)o.micro_candidates);

    const auto queries = GenMetricData(o, o.micro_queries, 3);
    std::mt19937_64 rng(4);
    std::uniform_int_distribution<faiss::idx_t> id_dist(0, o.nb - 1);
    std::vector<faiss::idx_t> ids(o.micro_candidates);
    for (auto& id : ids) {
        id = id_dist(rng);
    }

    const auto kernels = AvailableKernels(*index.aq);
    const size_t n_out = size_t(o.micro_queries) * ids.size();
    std::vector<float> ref_out;
    double ref_single = 0;

    std::printf("  %-12s %12s %12s %10s %10s %12s %12s %14s\n", "kernel", "single ns/c", "batch4 ns/c", "single x",
                "batch4 x", "LUT us/q", "bitwise eq", "max rel err");
    for (auto kernel : kernels) {
        faiss::set_aq_lut_kernel(kernel);
        std::unique_ptr<faiss::FlatCodesDistanceComputer> dc(index.get_FlatCodesDistanceComputer());
        std::vector<float> out(n_out);
        std::vector<double> t_single, t_batch, t_lut;
        double checksum = 0;

        for (int r = 0; r < o.repeats; r++) {
            double single = 0, batch = 0, lut = 0;
            for (int q = 0; q < o.micro_queries; q++) {
                auto t = Clock::now();
                dc->set_query(queries.data() + size_t(q) * o.dim);
                lut += SecondsSince(t);

                float* outq = out.data() + size_t(q) * ids.size();
                t = Clock::now();
                for (size_t i = 0; i < ids.size(); i++) {
                    outq[i] = (*dc)(ids[i]);
                }
                single += SecondsSince(t);
                for (size_t i = 0; i < ids.size(); i++) {
                    checksum += outq[i];
                }

                t = Clock::now();
                for (size_t i = 0; i < ids.size(); i += 4) {
                    dc->distances_batch_4(ids[i], ids[i + 1], ids[i + 2], ids[i + 3], outq[i], outq[i + 1], outq[i + 2],
                                          outq[i + 3]);
                }
                batch += SecondsSince(t);
                for (size_t i = 0; i < ids.size(); i++) {
                    checksum += outq[i];
                }
            }
            t_single.push_back(single * 1e9 / n_out);
            t_batch.push_back(batch * 1e9 / n_out);
            t_lut.push_back(lut * 1e6 / o.micro_queries);
        }

        auto s = Summarize(t_single);
        auto b = Summarize(t_batch);
        auto l = Summarize(t_lut);
        if (kernel == faiss::AQLutKernel::GENERIC) {
            ref_out = out;
            ref_single = s.median;
        }
        size_t bitwise_eq = 0;
        double max_rel = 0;
        for (size_t i = 0; i < n_out; i++) {
            bitwise_eq += std::memcmp(&out[i], &ref_out[i], sizeof(float)) == 0;
            max_rel =
                std::max(max_rel, std::abs(double(out[i]) - ref_out[i]) / std::max(1e-6, std::abs(double(ref_out[i]))));
        }
        std::printf("  %-12s %7.2f±%-4.2f %7.2f±%-4.2f %9.2fx %9.2fx %12.1f %11.2f%% %14.3g   (checksum %.6g)\n",
                    faiss::aq_lut_kernel_name(kernel), s.median, (s.max - s.min) / 2, b.median, (b.max - b.min) / 2,
                    ref_single / s.median, ref_single / b.median, l.median, 100.0 * bitwise_eq / n_out, max_rel,
                    checksum);
    }
    std::printf("  (ns/c = nanoseconds per candidate, median of %d runs ± half range; x = speedup vs generic single)\n",
                o.repeats);
}

double
Recall(const int64_t* gt, const int64_t* res, int64_t nq, int topk);

/*****************************************************************************
 * HNSW search with prebuilt LUTs
 *****************************************************************************/

struct NoopVisitor {
    using storage_idx_t = faiss::HNSW::storage_idx_t;
    void
    visit_level(const int) {
    }
    void
    visit_edge(const int, const storage_idx_t, const storage_idx_t, const float) {
    }
};

double
Percentile(std::vector<double>& v, double p) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(p * v.size()))];
}

void
RunHnswCachedLut(const Options& o, const PrqFixture& f) {
    using Searcher =
        faiss::cppcontrib::knowhere::v2_hnsw_searcher<faiss::DistanceComputer, NoopVisitor,
                                                      faiss::cppcontrib::knowhere::Bitset, faiss::IDSelectorAll>;
    std::printf("\n=== hnsw (prebuilt LUTs): M=%d efConstruction=%d, ef=%d, topk=%d, nq=%ld x %d passes ===\n",
                o.hnsw_m, o.ef_construction, o.ef, o.topk, (long)o.nq, o.hnsw_passes);

    // like knowhere HNSW_PRQ: the graph is built over the raw vectors, then searched over the PRQ codes
    faiss::IndexHNSWFlat graph(o.dim, o.hnsw_m, f.is_ip ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2);
    graph.hnsw.efConstruction = o.ef_construction;
    graph.hnsw.efSearch = o.ef;
    auto t0 = Clock::now();
    graph.add(o.nb, f.base.data());
    std::printf("  graph build: %.1fs\n", SecondsSince(t0));

    const auto queries = GenMetricData(o, o.nq, 5);
    std::vector<float> gt_dis(o.nq * o.topk);
    std::vector<faiss::idx_t> gt_ids(o.nq * o.topk);
    graph.storage->search(o.nq, queries.data(), o.topk, gt_dis.data(), gt_ids.data());

    const auto kernels = AvailableKernels(*f.index->aq);
    std::vector<faiss::idx_t> ref_ids;
    std::vector<double> ref_qps(o.hnsw_threads.size(), 0);

    std::printf("  %-12s %8s %10s %8s %10s %10s %10s %10s %10s\n", "kernel", "threads", "QPS", "x", "p50 us", "p95 us",
                "p99 us", "recall", "same ids");
    for (auto kernel : kernels) {
        faiss::set_aq_lut_kernel(kernel);

        // untimed: one distance computer per query, with its LUT
        t0 = Clock::now();
        std::vector<std::unique_ptr<faiss::DistanceComputer>> dcs(o.nq);
        for (int64_t q = 0; q < o.nq; q++) {
            faiss::DistanceComputer* dc = f.index->get_distance_computer();
            if (f.is_ip) {
                dc = new faiss::NegativeDistanceComputer(dc);
            }
            dc->set_query(queries.data() + q * o.dim);
            dcs[q].reset(dc);
        }
        const double lut_s = SecondsSince(t0);

        std::vector<faiss::idx_t> ids(o.nq * o.topk);
        std::vector<float> dis(o.nq * o.topk);
        for (size_t ti = 0; ti < o.hnsw_threads.size(); ti++) {
            const int nt = o.hnsw_threads[ti];
            const int64_t n_tasks = o.nq * o.hnsw_passes;
            std::vector<double> qps;
            std::vector<double> lat;
            for (int r = 0; r < o.repeats; r++) {
                std::atomic<int64_t> next_task{0};
                std::vector<std::vector<double>> lat_t(nt);
                auto worker = [&](int t) {
                    auto visited = faiss::cppcontrib::knowhere::Bitset::create_uninitialized(o.nb);
                    std::vector<faiss::idx_t> scratch_ids(o.topk);
                    std::vector<float> scratch_dis(o.topk);
                    NoopVisitor visitor;
                    faiss::IDSelectorAll sel;
                    for (int64_t task = next_task++; task < n_tasks; task = next_task++) {
                        const int64_t q = task % o.nq;
                        // only the first pass writes results, so no two threads write the same query
                        const bool first = task < o.nq;
                        auto tq = Clock::now();
                        visited.clear();
                        Searcher searcher{graph.hnsw, *dcs[q], visitor, visited, sel, 0.0f, nullptr};
                        searcher.search(o.topk, first ? dis.data() + q * o.topk : scratch_dis.data(),
                                        first ? ids.data() + q * o.topk : scratch_ids.data());
                        lat_t[t].push_back(SecondsSince(tq) * 1e6);
                    }
                };
                std::vector<std::thread> threads;
                t0 = Clock::now();
                for (int t = 0; t < nt; t++) {
                    threads.emplace_back(worker, t);
                }
                for (auto& th : threads) {
                    th.join();
                }
                qps.push_back(n_tasks / SecondsSince(t0));
                for (auto& l : lat_t) {
                    lat.insert(lat.end(), l.begin(), l.end());
                }
            }

            if (kernel == faiss::AQLutKernel::GENERIC) {
                ref_qps[ti] = Summarize(qps).median;
                if (ref_ids.empty()) {
                    ref_ids = ids;
                }
            }
            size_t same = 0;
            for (size_t i = 0; i < ids.size(); i++) {
                same += ids[i] == ref_ids[i];
            }
            auto s = Summarize(qps);
            std::printf("  %-12s %8d %10.0f %7.2fx %10.1f %10.1f %10.1f %10.4f %9.2f%%   (QPS range %.0f-%.0f)\n",
                        faiss::aq_lut_kernel_name(kernel), nt, s.median, s.median / ref_qps[ti], Percentile(lat, 0.5),
                        Percentile(lat, 0.95), Percentile(lat, 0.99), Recall(gt_ids.data(), ids.data(), o.nq, o.topk),
                        100.0 * same / ids.size(), s.min, s.max);
        }
        std::printf("  %-12s LUT build (untimed above): %.1f us/query\n", faiss::aq_lut_kernel_name(kernel),
                    lut_s * 1e6 / o.nq);
    }
}

/*****************************************************************************
 * end-to-end knowhere HNSW_PRQ search
 *****************************************************************************/

double
Recall(const int64_t* gt, const int64_t* res, int64_t nq, int topk) {
    int64_t hit = 0;
    for (int64_t q = 0; q < nq; q++) {
        std::unordered_set<int64_t> g(gt + q * topk, gt + (q + 1) * topk);
        for (int k = 0; k < topk; k++) {
            hit += g.count(res[q * topk + k]);
        }
    }
    return double(hit) / (nq * topk);
}

knowhere::KnowhereConfig::AQDistanceKernel
ToKnowhere(faiss::AQLutKernel k) {
    switch (k) {
        case faiss::AQLutKernel::GENERIC:
            return knowhere::KnowhereConfig::AQ_GENERIC;
        case faiss::AQLutKernel::SCALAR_8BIT:
            return knowhere::KnowhereConfig::AQ_SCALAR;
        case faiss::AQLutKernel::AVX512_8BIT:
            return knowhere::KnowhereConfig::AQ_AVX512;
        default:
            return knowhere::KnowhereConfig::AQ_AUTO;
    }
}

void
RunE2E(const Options& o) {
    std::printf(
        "\n=== e2e: HNSW_PRQ m=%d nrq=%d nbits=8, M=%d efConstruction=%d, ef=%d, topk=%d, dim=%d, nb=%ld, "
        "nq=%ld, metric=%s ===\n",
        o.prq_m, o.prq_nrq, o.hnsw_m, o.ef_construction, o.ef, o.topk, o.dim, (long)o.nb, (long)o.nq, o.metric.c_str());

    auto base_ds = ToDataSet(GenData(o.nb, o.dim, 2), o.nb, o.dim);
    auto query_vec = GenData(o.nq, o.dim, 5);
    auto query_ds = ToDataSet(query_vec, o.nq, o.dim);

    knowhere::Json json;
    json[knowhere::meta::DIM] = o.dim;
    json[knowhere::meta::METRIC_TYPE] = o.metric;
    json[knowhere::meta::TOPK] = o.topk;
    json[knowhere::indexparam::HNSW_M] = o.hnsw_m;
    json[knowhere::indexparam::EFCONSTRUCTION] = o.ef_construction;
    json[knowhere::indexparam::EF] = o.ef;
    json[knowhere::indexparam::M] = o.prq_m;
    json[knowhere::indexparam::PRQ_NUM] = o.prq_nrq;
    json[knowhere::indexparam::NBITS] = 8;

    auto t0 = Clock::now();
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(base_ds, query_ds, json, nullptr);
    if (!gt.has_value()) {
        std::fprintf(stderr, "brute force failed\n");
        std::exit(1);
    }
    std::printf("  ground truth: %.1fs\n", SecondsSince(t0));

    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_PRQ,
                                             knowhere::Version::GetCurrentVersion().VersionNumber())
                     .value();
    t0 = Clock::now();
    if (index.Build(base_ds, json) != knowhere::Status::success) {
        std::fprintf(stderr, "build failed\n");
        std::exit(1);
    }
    std::printf("  build: %.1fs\n", SecondsSince(t0));

    // kernel availability does not depend on the trained codebooks, only on nbits and the CPU
    faiss::ProductResidualQuantizer probe(o.dim, o.prq_m, o.prq_nrq, 8);
    const auto kernels = AvailableKernels(probe);

    std::vector<knowhere::DataSetPtr> single_queries;
    for (int64_t q = 0; q < o.nq; q++) {
        auto* data = new float[o.dim];
        std::memcpy(data, query_vec.data() + q * o.dim, sizeof(float) * o.dim);
        single_queries.push_back(knowhere::GenDataSet(1, o.dim, data));
    }

    std::printf("  %-12s %10s %8s %10s %10s %10s %10s %12s\n", "kernel", "QPS", "x", "p50 us", "p95 us", "p99 us",
                "recall", "same ids");
    std::vector<int64_t> ref_ids;
    double ref_qps = 0;
    for (auto kernel : kernels) {
        knowhere::KnowhereConfig::SetAQDistanceKernel(ToKnowhere(kernel));

        // batched search over all queries, parallel over the search thread pool
        std::vector<double> qps;
        knowhere::DataSetPtr res;
        for (int r = 0; r < o.repeats; r++) {
            t0 = Clock::now();
            auto rr = index.Search(query_ds, json, nullptr);
            qps.push_back(o.nq / SecondsSince(t0));
            res = rr.value();
        }
        // one query per call for latency percentiles
        std::vector<double> lat;
        for (int r = 0; r < std::max(1, o.repeats / 2); r++) {
            for (auto& q : single_queries) {
                t0 = Clock::now();
                auto rr = index.Search(q, json, nullptr);
                lat.push_back(SecondsSince(t0) * 1e6);
            }
        }
        std::sort(lat.begin(), lat.end());
        auto pct = [&](double p) { return lat[std::min(lat.size() - 1, size_t(p * lat.size()))]; };

        const int64_t* ids = res->GetIds();
        std::vector<int64_t> ids_vec(ids, ids + o.nq * o.topk);
        if (kernel == faiss::AQLutKernel::GENERIC) {
            ref_ids = ids_vec;
            ref_qps = Summarize(qps).median;
        }
        size_t same = 0;
        for (size_t i = 0; i < ids_vec.size(); i++) {
            same += ids_vec[i] == ref_ids[i];
        }
        auto s = Summarize(qps);
        std::printf("  %-12s %10.0f %7.2fx %10.1f %10.1f %10.1f %10.4f %11.2f%%   (QPS range %.0f-%.0f)\n",
                    faiss::aq_lut_kernel_name(kernel), s.median, s.median / ref_qps, pct(0.5), pct(0.95), pct(0.99),
                    Recall(gt.value()->GetIds(), ids, o.nq, o.topk), 100.0 * same / ids_vec.size(), s.min, s.max);
    }
    knowhere::KnowhereConfig::SetAQDistanceKernel(knowhere::KnowhereConfig::AQ_AUTO);
}

}  // namespace

int
main(int argc, char** argv) {
    Options o;
    if (!ParseArgs(argc, argv, o)) {
        return 1;
    }
#if defined(__x86_64__)
    std::printf("cpu avx512 support: %d\n", (int)faiss::cpu_support_avx512());
#endif
    if (o.search_threads > 0) {
        knowhere::KnowhereConfig::SetSearchThreadPoolSize(o.search_threads);
    }
    if (o.run_micro || o.run_hnsw) {
        const auto prq = BuildPrq(o);
        if (o.run_micro) {
            RunMicro(o, prq);
        }
        if (o.run_hnsw) {
            RunHnswCachedLut(o, prq);
        }
    }
    if (o.run_e2e) {
        RunE2E(o);
    }
    return 0;
}

/* Acfun GEMM — AVX-512F + FMA 后端 (GotoBLAS 风格三级分块)
 *
 * 关键优化 (对标 MNN / 一线 BLAS 的工程手段):
 *   1) 数据打包: A 面板按 k 连续重排 ([kc*MC + i]); B 面板同样按 k 连续重排到
 *      [kc*NC + j] 的连续缓冲区, 使微内核沿 k 的 B 访问完全顺序化, 彻底消除
 *      原实现对 B 的 stride-N 跨步访问, B 工作集常驻 L2。
 *   2) 三级分块: MC(命中 L3) × KC(命中 L2) × NC(命中 L1), 微内核工作集 (MR+NR)*KC
 *      仅 ~(MR+NR)*KC*4 字节, 稳稳落在 L1, 算术强度高。
 *   3) 并行 N 块: OpenMP 并行外层 N 维, 每个 N 块的 B 面板被该线程内所有 M 块复用,
 *      提升缓存复用与负载均衡。
 *   4) β=1 跨 KC 累加: C 的同一 (M,N) 瓦片跨多个 KC 块累加, k0==0 块以 β=0 覆盖、
 *      其余以 β=1 累加, 单精度无误差累积。
 *
 * 微内核只处理 [0,mrR)×[0,nrR) 的完整 16×16 块; 行/列尾部余数交给标量参考,
 * 余数直接用原始 A/B (k 主序面板与参考的 i 主序读取不兼容, 故用原矩阵), 保证正确。
 *
 * 编译单元需带 -mavx512f -mfma。 */
#include "internal.h"
#include <immintrin.h>
#include <omp.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <unistd.h>

static int g_dim(const char* env, int def) {
    const char* e = std::getenv(env);
    if (e) { int v = std::atoi(e); if (v > 0) return v; }
    return def;
}

/* 硬件自适应分块: 未显式指定时, 按 L2 容量选取 KC/NC, 使 B 面板(KC*NC*4) 与
 * A 面板(MC*KC*4) 合计稳居 L2 (MC 固定 256)。对标 MNN 的按硬件调参。 */
static void auto_blocks(int* KC, int* NC) {
    long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    long cap = (l2 > 0) ? l2 / 2 : (1L << 19);   /* 取 L2 一半留余量 */
    int kc = 256, nc = 256;
    if ((long)kc * nc * 4 + (long)256 * kc * 4 > cap) { kc = 128; nc = 128; }
    if ((long)kc * nc * 4 + (long)256 * kc * 4 > cap) { kc = 64;  nc = 64;  }
    *KC = kc; *NC = nc;
}

/* 微内核: 仅完整 16×16 块, C 瓦片 += A_panel * B_panel; beta 控制种子 (0=覆盖/1=累加) */
static inline void kernel_16x16(const float* Ap, const float* Bp,
                                float* C, int ldC,
                                int mc, int nc, int kc,
                                int MC, int NC, int beta) {
    int mrR = mc - (mc & 15);
    int nrR = nc - (nc & 15);
    for (int mm = 0; mm < mrR; mm += 16) {
        for (int nn = 0; nn < nrR; nn += 16) {
            __m512 c[16];
            if (beta == 0) {
                for (int i = 0; i < 16; ++i) c[i] = _mm512_setzero_ps();
            } else {
                for (int i = 0; i < 16; ++i)
                    c[i] = _mm512_loadu_ps(C + ((size_t)(mm + i)) * ldC + nn);
            }
            for (int k = 0; k < kc; ++k) {
                __m512 b = _mm512_loadu_ps(Bp + ((size_t)k) * NC + nn);
                for (int i = 0; i < 16; ++i) {
                    float a = Ap[((size_t)k) * MC + mm + i];
                    c[i] = _mm512_fmadd_ps(_mm512_set1_ps(a), b, c[i]);
                }
                _mm_prefetch((const char*)(Bp + ((size_t)(k + 16)) * NC + nn), _MM_HINT_T0);
                _mm_prefetch((const char*)(Ap + ((size_t)(k + 16)) * MC + mm), _MM_HINT_T0);
            }
            for (int i = 0; i < 16; ++i)
                _mm512_storeu_ps(C + ((size_t)(mm + i)) * ldC + nn, c[i]);
        }
    }
}

void acfun_gemm_avx512(int M, int N, int K,
                       const float* A, const float* B, float* C) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    const int MR = 16, NR = 16;
    int MC = 256;
    int KC = 256, NC = 256;
    if (std::getenv("ACFUN_KC") || std::getenv("ACFUN_NC")) {
        KC = g_dim("ACFUN_KC", 256);
        NC = g_dim("ACFUN_NC", 256);
    } else {
        auto_blocks(&KC, &NC);
    }
    MC = (MC / MR) * MR; if (MC < MR) MC = MR;
    NC = (NC / NR) * NR; if (NC < NR) NC = NR;
    if (KC < 8) KC = 8;
    const int lda = K, ldb = N, ldc = N;

    #pragma omp parallel
    {
        float* Ap = (float*)malloc((size_t)MC * KC * sizeof(float));
        float* Bp = (float*)malloc((size_t)KC * NC * sizeof(float));
        if (!Ap || !Bp) { free(Ap); free(Bp); goto end_par; }

        #pragma omp for schedule(static, 1)
        for (int n0 = 0; n0 < N; n0 += NC) {
            int nc = std::min(NC, N - n0);
            for (int k0 = 0; k0 < K; k0 += KC) {
                int kc = std::min(KC, K - k0);
                int beta = (k0 == 0) ? 0 : 1;   /* 同一 (m0,n0) 瓦片跨 KC 块累加 */

                /* 打包 B 面板: Bp[k*NC + j] = B[(k0+k)*ldb + (n0+j)], j<nc (连续行, memcpy) */
                for (int k = 0; k < kc; ++k)
                    std::memcpy(Bp + (size_t)k * NC,
                                B + ((size_t)(k0 + k)) * ldb + n0,
                                (size_t)nc * sizeof(float));

                for (int m0 = 0; m0 < M; m0 += MC) {
                    int mc = std::min(MC, M - m0);
                    /* 打包 A 面板: Ap[k*MC + i] = A[(m0+i)*lda + (k0+k)], i<mc */
                    for (int k = 0; k < kc; ++k)
                        for (int i = 0; i < mc; ++i)
                            Ap[((size_t)k) * MC + i] = A[((size_t)(m0 + i)) * lda + (k0 + k)];

                    kernel_16x16(Ap, Bp, C + (size_t)m0 * ldc + n0, ldc,
                                 mc, nc, kc, MC, NC, beta);

                    /* 尾部余数: 直接用原始 A/B (beta 与全块一致) */
                    int mrR = mc - (mc & 15), nrR = nc - (nc & 15);
                    if (mc > mrR) {
                        auto f = beta ? acfun_gemm_ref_ld_b1 : acfun_gemm_ref_ld;
                        f(mc - mrR, nc, kc, lda, ldb, ldc,
                          A + ((size_t)(m0 + mrR)) * lda + k0,
                          B + ((size_t)k0) * ldb + n0,
                          C + ((size_t)(m0 + mrR)) * ldc + n0);
                    }
                    if (nc > nrR) {
                        auto f = beta ? acfun_gemm_ref_ld_b1 : acfun_gemm_ref_ld;
                        f(mrR, nc - nrR, kc, lda, ldb, ldc,
                          A + ((size_t)m0) * lda + k0,
                          B + ((size_t)k0) * ldb + n0 + nrR,
                          C + ((size_t)m0) * ldc + n0 + nrR);
                    }
                }
            }
        }
        end_par:
        free(Ap); free(Bp);
    }
}

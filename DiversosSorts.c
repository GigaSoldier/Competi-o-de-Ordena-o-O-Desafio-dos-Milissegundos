#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>
#if defined(_MSC_VER)
#include <malloc.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_thread_num(void)  { return 0; }
static inline int omp_get_num_threads(void) { return 1; }
static inline int omp_get_max_threads(void) { return 1; }
static inline int omp_get_num_procs(void)   { return 1; }
static inline void omp_set_num_threads(int n) { (void)n; }
static inline void omp_set_dynamic(int d)     { (void)d; }
#endif

/* SIMD selecionado na compilacao para a arquitetura-alvo. */
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX512F__)
#  include <immintrin.h>
#  define USA_AVX512 1
#  define USA_AVX2 1 /* AVX-512 e retrocompativel com os caminhos AVX2 existentes. */
#elif (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
#  include <immintrin.h>
#  define USA_AVX2 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#  include <arm_neon.h>
#  define USA_NEON 1
#endif

typedef int32_t value_t;

#define TROCA(a, b) do { value_t t_ = (a); (a) = (b); (b) = t_; } while (0)
#define MAX_T 256
#define MAX_SEG 1024
#define ORCAMENTO_HIST ((size_t)512 << 20)
#define REPETICOES 5

typedef struct {
    const char *nome;
    double clock_ms;
    double parede_ms;
    int aplicavel;
    int correto;
} Resultado;

typedef struct {
    uint32_t *hist;
    size_t cap;
} Workspace;

typedef struct {
    uint64_t soma, soma2;
} Assinatura;

/* ------------------------------------------------------------------ */
/* Gerenciamento de Memória Alinhada em Cache Line (64 Bytes)         */
/* ------------------------------------------------------------------ */

static void *malloc_alinhado(size_t bytes) {
    void *ptr = NULL;
#if defined(_MSC_VER)
    ptr = _aligned_malloc(bytes, 64);
#elif defined(_POSIX_C_SOURCE) && (_POSIX_C_SOURCE >= 200112L) || defined(__APPLE__) || defined(__linux__)
    if (posix_memalign(&ptr, 64, bytes) != 0) ptr = NULL;
#else
    ptr = malloc(bytes);
#endif
    return ptr;
}

static void free_alinhado(void *ptr) {
    if (!ptr) return;
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static inline uint32_t mapToUnsigned(value_t val) {
    return (uint32_t)val ^ 0x80000000U;
}

static inline value_t mapFromUnsigned(uint32_t val) {
    return (value_t)(val ^ 0x80000000U);
}

static double agora_ms(void) {
#ifdef _OPENMP
    return omp_get_wtime() * 1000.0;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
#endif
}

/* Mostra recursos reais da CPU em tempo de execucao (x86 GCC/Clang). */
static void imprimir_capacidades_cpu(void) {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    printf(" CPU (CPUID):");
    int algum = 0;
    if (__builtin_cpu_supports("avx512f"))  { printf(" AVX-512F"); algum = 1; }
    if (__builtin_cpu_supports("avx512bw")) { printf(" AVX-512BW"); algum = 1; }
    if (__builtin_cpu_supports("avx512dq")) { printf(" AVX-512DQ"); algum = 1; }
    if (__builtin_cpu_supports("avx512vl")) { printf(" AVX-512VL"); algum = 1; }
    if (!algum && __builtin_cpu_supports("avx2")) { printf(" AVX2"); algum = 1; }
    if (!algum) printf(" sem AVX2/AVX-512");
    printf("\n");
#elif defined(__aarch64__) || defined(_M_ARM64)
    printf(" CPU: ARM64 (Advanced SIMD/NEON obrigatorio)\n");
#endif
}

static inline int bits_de(uint32_t x) {
#if defined(__GNUC__)
    return x ? 32 - __builtin_clz(x) : 0;
#else
    int b = 0;
    for (; x; x >>= 1) ++b;
    return b;
#endif
}

static size_t particao(size_t n, int t, int nt) {
    return (size_t)((unsigned long long)n * (unsigned long long)t / (unsigned long long)nt);
}

static int threads_para(size_t n) {
    int T = omp_get_max_threads();
    size_t lim = n >> 15;
    if (lim < 1) lim = 1;
    if ((size_t)T > lim) T = (int)lim;
    if (T > MAX_T) T = MAX_T;
    if (T < 1) T = 1;
    return T;
}

static uint32_t *ws_hist(Workspace *ws, size_t elementos) {
    if (ws->cap < elementos) {
        free_alinhado(ws->hist);
        ws->hist = (uint32_t *)malloc_alinhado(elementos * sizeof(uint32_t));
        ws->cap = ws->hist ? elementos : 0;
    }
    return ws->hist;
}

static Assinatura assinatura(const value_t *v, size_t n) {
    uint64_t s1 = 0, s2 = 0;
    long long cnt = (long long)n;
#pragma omp parallel for reduction(+:s1,s2) schedule(static) if(cnt > 100000)
    for (long long i = 0; i < cnt; ++i) {
        uint64_t x = (uint64_t)(int64_t)v[i];
        s1 += x;
        s2 += x * x;
    }
    return (Assinatura){s1, s2};
}

static int bloco_ordenado(const value_t *v, size_t a, size_t b) {
    size_t i = a;
#ifdef USA_AVX512
    for (; i + 16 <= b; i += 16) {
        __m512i x = _mm512_loadu_si512((const void *)(v + i - 1));
        __m512i y = _mm512_loadu_si512((const void *)(v + i));
        if (_mm512_cmpgt_epi32_mask(x, y)) return 0;
    }
#elif defined(USA_AVX2)
    for (; i + 8 <= b; i += 8) {
        __m256i x = _mm256_loadu_si256((const __m256i *)(v + i - 1));
        __m256i y = _mm256_loadu_si256((const __m256i *)(v + i));
        if (_mm256_movemask_epi8(_mm256_cmpgt_epi32(x, y))) return 0;
    }
#elif defined(USA_NEON)
    for (; i + 4 <= b; i += 4) {
        int32x4_t x = vld1q_s32(v + i - 1);
        int32x4_t y = vld1q_s32(v + i);
        if (vminvq_u32(vcleq_s32(x, y)) == 0) return 0;
    }
#endif
    for (; i < b; ++i)
        if (v[i - 1] > v[i]) return 0;
    return 1;
}

static int esta_ordenado(const value_t *v, size_t n) {
    int ok = 1;
    int T = threads_para(n);
#pragma omp parallel num_threads(T) reduction(&:ok)
    {
        int t = omp_get_thread_num(), nt = omp_get_num_threads();
        size_t a = particao(n, t, nt), b = particao(n, t + 1, nt);
        if (a < 1) a = 1;
        ok &= bloco_ordenado(v, a, b);
    }
    return ok;
}

/* Força 100% de uso da CPU antes de cronometrar (Locks de Turbo Boost e aquecimento de L1/L2) */
static void aquecer_cpu_e_threads(void) {
    int T = omp_get_max_threads();
#pragma omp parallel num_threads(T)
    {
        volatile double acc = 1.0000001;
        for (int i = 0; i < 2000000; i++) {
            acc = acc * 1.0000001 + 0.0000001;
        }
        (void)acc;
    }
}

static void copia_paralela(value_t *dst, const value_t *src, size_t n, int T) {
#pragma omp parallel num_threads(T)
    {
        int t = omp_get_thread_num(), nt = omp_get_num_threads();
        size_t a = particao(n, t, nt), b = particao(n, t + 1, nt);
        memcpy(dst + a, src + a, (b - a) * sizeof(value_t));
    }
}

static void minmax_bloco(const value_t *v, size_t a, size_t b, value_t *lo_out, value_t *hi_out) {
    value_t lo = INT_MAX, hi = INT_MIN;
    size_t i = a;
#ifdef USA_AVX512
    if (b - a >= 64) {
        __m512i vlo0 = _mm512_set1_epi32(INT_MAX), vlo1 = vlo0;
        __m512i vhi0 = _mm512_set1_epi32(INT_MIN), vhi1 = vhi0;
        for (; i + 64 <= b; i += 64) {
            __m512i x0 = _mm512_loadu_si512((const void *)(v + i));
            __m512i x1 = _mm512_loadu_si512((const void *)(v + i + 16));
            __m512i x2 = _mm512_loadu_si512((const void *)(v + i + 32));
            __m512i x3 = _mm512_loadu_si512((const void *)(v + i + 48));
            vlo0 = _mm512_min_epi32(vlo0, _mm512_min_epi32(x0, x1));
            vlo1 = _mm512_min_epi32(vlo1, _mm512_min_epi32(x2, x3));
            vhi0 = _mm512_max_epi32(vhi0, _mm512_max_epi32(x0, x1));
            vhi1 = _mm512_max_epi32(vhi1, _mm512_max_epi32(x2, x3));
        }
        lo = _mm512_reduce_min_epi32(_mm512_min_epi32(vlo0, vlo1));
        hi = _mm512_reduce_max_epi32(_mm512_max_epi32(vhi0, vhi1));
    }
#elif defined(USA_AVX2)
    if (b - a >= 32) {
        __m256i l0 = _mm256_set1_epi32(INT_MAX), l1 = l0, l2 = l0, l3 = l0;
        __m256i h0 = _mm256_set1_epi32(INT_MIN), h1 = h0, h2 = h0, h3 = h0;
        for (; i + 32 <= b; i += 32) {
            __m256i x0 = _mm256_loadu_si256((const __m256i *)(v + i));
            __m256i x1 = _mm256_loadu_si256((const __m256i *)(v + i + 8));
            __m256i x2 = _mm256_loadu_si256((const __m256i *)(v + i + 16));
            __m256i x3 = _mm256_loadu_si256((const __m256i *)(v + i + 24));
            l0 = _mm256_min_epi32(l0, x0); h0 = _mm256_max_epi32(h0, x0);
            l1 = _mm256_min_epi32(l1, x1); h1 = _mm256_max_epi32(h1, x1);
            l2 = _mm256_min_epi32(l2, x2); h2 = _mm256_max_epi32(h2, x2);
            l3 = _mm256_min_epi32(l3, x3); h3 = _mm256_max_epi32(h3, x3);
        }
        l0 = _mm256_min_epi32(_mm256_min_epi32(l0, l1), _mm256_min_epi32(l2, l3));
        h0 = _mm256_max_epi32(_mm256_max_epi32(h0, h1), _mm256_max_epi32(h2, h3));
        __m128i l = _mm_min_epi32(_mm256_castsi256_si128(l0), _mm256_extracti128_si256(l0, 1));
        __m128i h = _mm_max_epi32(_mm256_castsi256_si128(h0), _mm256_extracti128_si256(h0, 1));
        l = _mm_min_epi32(l, _mm_shuffle_epi32(l, _MM_SHUFFLE(1, 0, 3, 2)));
        l = _mm_min_epi32(l, _mm_shuffle_epi32(l, _MM_SHUFFLE(2, 3, 0, 1)));
        h = _mm_max_epi32(h, _mm_shuffle_epi32(h, _MM_SHUFFLE(1, 0, 3, 2)));
        h = _mm_max_epi32(h, _mm_shuffle_epi32(h, _MM_SHUFFLE(2, 3, 0, 1)));
        lo = _mm_cvtsi128_si32(l);
        hi = _mm_cvtsi128_si32(h);
    }
#elif defined(USA_NEON)
    if (b - a >= 16) {
        int32x4_t vlo = vdupq_n_s32(INT_MAX);
        int32x4_t vhi = vdupq_n_s32(INT_MIN);
        for (; i + 16 <= b; i += 16) {
            int32x4_t x0 = vld1q_s32(v + i);
            int32x4_t x1 = vld1q_s32(v + i + 4);
            int32x4_t x2 = vld1q_s32(v + i + 8);
            int32x4_t x3 = vld1q_s32(v + i + 12);
            vlo = vminq_s32(vlo, vminq_s32(vminq_s32(x0, x1), vminq_s32(x2, x3)));
            vhi = vmaxq_s32(vhi, vmaxq_s32(vmaxq_s32(x0, x1), vmaxq_s32(x2, x3)));
        }
        lo = vminvq_s32(vlo);
        hi = vmaxvq_s32(vhi);
    }
#endif
    for (; i < b; ++i) {
        value_t x = v[i];
        lo = x < lo ? x : lo;
        hi = x > hi ? x : hi;
    }
    *lo_out = lo;
    *hi_out = hi;
}

static void minmax_par(const value_t *v, size_t n, int T, value_t *mn, value_t *mx) {
    value_t mins[MAX_T], maxs[MAX_T];
    int usados = 1;
#pragma omp parallel num_threads(T)
    {
        int t = omp_get_thread_num(), nt = omp_get_num_threads();
        size_t a = particao(n, t, nt), b = particao(n, t + 1, nt);
        value_t lo, hi;
        minmax_bloco(v, a, b, &lo, &hi);
        mins[t] = lo;
        maxs[t] = hi;
        if (t == 0) usados = nt;
    }
    value_t lo = mins[0], hi = maxs[0];
    for (int k = 1; k < usados; ++k) {
        if (mins[k] < lo) lo = mins[k];
        if (maxs[k] > hi) hi = maxs[k];
    }
    *mn = lo;
    *mx = hi;
}

/* Leitura paralela acelerada do buffer de disco */
#define EH_TOKEN(c) ((((unsigned)(c) - '0') < 10u) || (c) == '-')

static size_t varrer(const unsigned char *buf, size_t lo, size_t hi, value_t *out) {
    const unsigned char *p = buf + lo, *e = buf + hi;
    size_t cont = 0;
    if (lo > 0 && EH_TOKEN(buf[lo - 1]))
        while (p < e && EH_TOKEN(*p)) ++p;
    while (p < e) {
        unsigned c = *p;
        if (EH_TOKEN(c)) {
            int neg = 0;
            long long x = 0;
            if (c == '-') { neg = 1; ++p; }
            while (((unsigned)(*p) - '0') < 10u) {
                x = x * 10 + (*p - '0');
                ++p;
            }
            if (out) out[cont] = (value_t)(neg ? -x : x);
            ++cont;
        } else {
            ++p;
        }
    }
    return cont;
}

static value_t *carregar_arquivo(const char *caminho, size_t *n_out) {
    FILE *f = fopen(caminho, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long long tam = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (tam <= 0) { fclose(f); return NULL; }

    unsigned char *buf = (unsigned char *)malloc_alinhado((size_t)tam + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t lidos = fread(buf, 1, (size_t)tam, f);
    fclose(f);
    buf[lidos] = 0;

    int T = (lidos < ((size_t)1 << 20)) ? 1 : omp_get_max_threads();
    if (T > MAX_T) T = MAX_T;

    size_t cnt[MAX_T + 1], off[MAX_T + 1];
    value_t *dados = NULL;
    size_t total = 0;

#pragma omp parallel num_threads(T)
    {
        int t = omp_get_thread_num(), nt = omp_get_num_threads();
        size_t lo = particao(lidos, t, nt), hi = particao(lidos, t + 1, nt);
        cnt[t] = varrer(buf, lo, hi, NULL);
#pragma omp barrier
#pragma omp single
        {
            size_t acc = 0;
            for (int k = 0; k < nt; ++k) { off[k] = acc; acc += cnt[k]; }
            total = acc;
            dados = (value_t *)malloc_alinhado((total ? total : 1) * sizeof(value_t));
        }
        if (dados) varrer(buf, lo, hi, dados + off[t]);
    }

    free_alinhado(buf);
    *n_out = dados ? total : 0;
    return dados;
}

/* ------------------------------------------------------------------ */
/* Algoritmos de Ordenação Otimizados                                 */
/* ------------------------------------------------------------------ */

static void insertion_sort(value_t *v, size_t inicio, size_t fim);
static void merge_seq(const value_t *A, size_t na, const value_t *B, size_t nb, value_t *out);

#define CORTE_INS 64

static void msort_pp(value_t *dst, value_t *src, size_t ini, size_t fim) {
    if (fim - ini <= CORTE_INS) {
        insertion_sort(dst, ini, fim);
        return;
    }

#if defined(USA_AVX2)
    size_t meio = ini + (((fim - ini) / 2 + 32) / 64) * 64;
#elif defined(USA_NEON)
    size_t meio = ini + (((fim - ini) / 2 + 2) / 4) * 4;
#else
    size_t meio = ini + (fim - ini) / 2;
#endif
    msort_pp(src, dst, ini, meio);
    msort_pp(src, dst, meio, fim);

    if (src[meio - 1] <= src[meio]) {
        memcpy(dst + ini, src + ini, (fim - ini) * sizeof(*dst));
        return;
    }

    size_t i = ini, j = meio, k = ini;
    while (i < meio && j < fim) {
        value_t a = src[i], b = src[j];
        int pega_a = (a <= b);
        dst[k++] = pega_a ? a : b;
        i += pega_a;
        j += !pega_a;
    }
    if (i < meio) memcpy(dst + k, src + i, (meio - i) * sizeof(*dst));
    else if (j < fim) memcpy(dst + k, src + j, (fim - j) * sizeof(*dst));
}

static void mergesort_rec(value_t *v, value_t *aux, size_t inicio, size_t fim) {
    if (fim - inicio < 2) return;
    memcpy(aux + inicio, v + inicio, (fim - inicio) * sizeof(*v));
    msort_pp(v, aux, inicio, fim);
}

static int comparar_int(const void *a, const void *b) {
    value_t x = *(const value_t *)a;
    value_t y = *(const value_t *)b;
    return (x > y) - (x < y);
}

static void insertion_sort(value_t *v, size_t inicio, size_t fim) {
    for (size_t i = inicio + 1; i < fim; ++i) {
        value_t chave = v[i];
        size_t j = i;
        while (j > inicio && v[j - 1] > chave) {
            v[j] = v[j - 1];
            --j;
        }
        v[j] = chave;
    }
}

static void sift_down(value_t *v, size_t inicio, size_t n) {
    size_t raiz = inicio;
    while (n >= 2 && raiz <= (n - 2) / 2) {
        size_t filho = raiz * 2 + 1;
        if (filho + 1 < n && v[filho] < v[filho + 1]) ++filho;
        if (v[raiz] >= v[filho]) return;
        TROCA(v[raiz], v[filho]);
        raiz = filho;
    }
}

static void heapsort_intervalo(value_t *v, size_t inicio, size_t fim) {
    size_t n = fim - inicio;
    if (n < 2) return;
    for (size_t i = n / 2; i > 0; --i)
        sift_down(v + inicio, i - 1, n);
    for (size_t fim_heap = n; fim_heap > 1; --fim_heap) {
        TROCA(v[inicio], v[inicio + fim_heap - 1]);
        sift_down(v + inicio, 0, fim_heap - 1);
    }
}

static size_t particionar(value_t *v, size_t inicio, size_t fim) {
    size_t meio = inicio + (fim - inicio) / 2;
    size_t ultimo = fim - 1;
    if (v[meio] < v[inicio])  TROCA(v[meio], v[inicio]);
    if (v[ultimo] < v[meio])  TROCA(v[ultimo], v[meio]);
    if (v[meio] < v[inicio])  TROCA(v[meio], v[inicio]);
    value_t pivo = v[meio];

    size_t i = inicio, j = fim - 1;
    for (;;) {
        while (v[i] < pivo) ++i;
        while (v[j] > pivo) --j;
        if (i >= j) break;
        TROCA(v[i], v[j]);
        ++i;
        if (j > 0) --j;
    }
    return j + 1;
}

static void introsort_rec(value_t *v, size_t inicio, size_t fim, int profundidade) {
    while (fim - inicio > 24) {
        if (profundidade == 0) {
            heapsort_intervalo(v, inicio, fim);
            return;
        }
        --profundidade;

        size_t corte = particionar(v, inicio, fim);

        if (corte - inicio < fim - corte) {
            introsort_rec(v, inicio, corte, profundidade);
            inicio = corte;
        } else {
            introsort_rec(v, corte, fim, profundidade);
            fim = corte;
        }
    }
    insertion_sort(v, inicio, fim);
}

static int log2_piso(size_t x) {
#if defined(__GNUC__)
    return x ? 63 - __builtin_clzll((unsigned long long)x) : 0;
#else
    int p = 0;
    for (; x > 1; x >>= 1) ++p;
    return p;
#endif
}

static void introsort(value_t *v, size_t n) {
    introsort_rec(v, 0, n, log2_piso(n) * 2);
}

static void introsort_paralelo(value_t *v, size_t n) {
    int T = threads_para(n);
    if (T == 1 || n < ((size_t)1 << 16)) { introsort(v, n); return; }

    size_t ini_a[MAX_SEG], fim_a[MAX_SEG], ini_b[MAX_SEG], fim_b[MAX_SEG];
    size_t *si = ini_a, *sf = fim_a, *di = ini_b, *df = fim_b;
    int m = 1;
    int alvo = 4 * T;
    if (alvo > MAX_SEG) alvo = MAX_SEG;
    si[0] = 0;
    sf[0] = n;

    while (m < alvo) {
        int thr = (m < T) ? m : T;
#pragma omp parallel for num_threads(thr) schedule(static)
        for (int s = 0; s < m; ++s) {
            size_t a = si[s], b = sf[s];
            if (b - a > 4096) {
                size_t c = particionar(v, a, b);
                di[2 * s] = a;     df[2 * s] = c;
                di[2 * s + 1] = c; df[2 * s + 1] = b;
            } else {
                di[2 * s] = a;     df[2 * s] = b;
                di[2 * s + 1] = b; df[2 * s + 1] = b;
            }
        }
        size_t *t1 = si; si = di; di = t1;
        size_t *t2 = sf; sf = df; df = t2;
        m *= 2;
    }

#pragma omp parallel for num_threads(T) schedule(dynamic, 1)
    for (int s = 0; s < m; ++s)
        introsort(v + si[s], sf[s] - si[s]);
}

static void hist_somar(uint32_t *dst, const uint32_t *src, size_t n) {
    size_t i = 0;
#ifdef USA_AVX512
    for (; i + 16 <= n; i += 16) {
        __m512i a = _mm512_loadu_si512((const void *)(dst + i));
        __m512i b = _mm512_loadu_si512((const void *)(src + i));
        _mm512_storeu_si512((void *)(dst + i), _mm512_add_epi32(a, b));
    }
#elif defined(USA_AVX2)
    for (; i + 8 <= n; i += 8) {
        __m256i a = _mm256_loadu_si256((const __m256i *)(dst + i));
        __m256i b = _mm256_loadu_si256((const __m256i *)(src + i));
        _mm256_storeu_si256((__m256i *)(dst + i), _mm256_add_epi32(a, b));
    }
#elif defined(USA_NEON)
    for (; i + 4 <= n; i += 4) {
        uint32x4_t a = vld1q_u32(dst + i);
        uint32x4_t b = vld1q_u32(src + i);
        vst1q_u32(dst + i, vaddq_u32(a, b));
    }
#endif
    for (; i < n; ++i) dst[i] += src[i];
}

static size_t hist_total(const uint32_t *h, size_t n) {
    size_t i = 0, s = 0;
#ifdef USA_AVX512
    __m512i acc = _mm512_setzero_si512();
    for (; i + 16 <= n; i += 16) {
        __m512i x = _mm512_loadu_si512((const void *)(h + i));
        acc = _mm512_add_epi32(acc, x);
    }
    /* O total e menor que UINT32_MAX, portanto a soma por lanes nao transborda. */
    s = (size_t)(uint32_t)_mm512_reduce_add_epi32(acc);
#elif defined(USA_AVX2)
    __m256i acc = _mm256_setzero_si256();
    for (; i + 8 <= n; i += 8) {
        __m256i x = _mm256_loadu_si256((const __m256i *)(h + i));
        acc = _mm256_add_epi64(acc, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(x)));
        acc = _mm256_add_epi64(acc, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(x, 1)));
    }
    uint64_t t[4];
    _mm256_storeu_si256((__m256i *)t, acc);
    s = (size_t)(t[0] + t[1] + t[2] + t[3]);
#elif defined(USA_NEON)
    for (; i + 4 <= n; i += 4) {
        uint32x4_t x = vld1q_u32(h + i);
        s += (size_t)vaddlvq_u32(x);
    }
#endif
    for (; i < n; ++i) s += h[i];
    return s;
}

static inline void preencher(value_t *dst, value_t val, size_t cnt) {
    size_t j = 0;
#ifdef USA_AVX512
    if (cnt >= 16) {
        __m512i vv = _mm512_set1_epi32(val);
        for (; j + 16 <= cnt; j += 16) _mm512_storeu_si512((void *)(dst + j), vv);
    }
#elif defined(USA_AVX2)
    if (cnt >= 8) {
        __m256i vv = _mm256_set1_epi32(val);
        for (; j + 8 <= cnt; j += 8) _mm256_storeu_si256((__m256i *)(dst + j), vv);
    }
#elif defined(USA_NEON)
    int32x4_t vv = vdupq_n_s32(val);
    for (; j + 4 <= cnt; j += 4) vst1q_s32(dst + j, vv);
#endif
    for (; j < cnt; ++j) dst[j] = val;
}

static int ordenar_contagem(value_t *v, size_t n, Workspace *ws) {
    if (n < 2) return 0;
    if (n >= 0xFFFFFFFFu) return -1;

    int T = threads_para(n);
    value_t mn, mx;
    minmax_par(v, n, T, &mn, &mx);

    uint64_t faixa = (uint64_t)((int64_t)mx - (int64_t)mn) + 1;
    if (faixa > 4 * (uint64_t)n + 65536) return -1;
    if (faixa * sizeof(uint32_t) * (uint64_t)T > ORCAMENTO_HIST) return -1;

    size_t R = (size_t)faixa;
    uint32_t *H = ws_hist(ws, (size_t)T * R);
    if (!H) return -1;

    uint32_t mnu = (uint32_t)mn;
    size_t fatia[MAX_T], desloc[MAX_T];

#pragma omp parallel num_threads(T)
    {
        int t = omp_get_thread_num(), nt = omp_get_num_threads();
        uint32_t *h = H + (size_t)t * R;
        memset(h, 0, R * sizeof(uint32_t));

        size_t a = particao(n, t, nt), b = particao(n, t + 1, nt);
        for (size_t i = a; i < b; ++i)
            h[(uint32_t)v[i] - mnu]++;
#pragma omp barrier

        size_t va = particao(R, t, nt), vb = particao(R, t + 1, nt);
        for (int k = 1; k < nt; ++k)
            hist_somar(H + va, H + (size_t)k * R + va, vb - va);
        fatia[t] = hist_total(H + va, vb - va);
#pragma omp barrier
#pragma omp single
        {
            size_t acc = 0;
            for (int k = 0; k < nt; ++k) { desloc[k] = acc; acc += fatia[k]; }
        }

        size_t pos = desloc[t];
        for (size_t x = va; x < vb; ++x) {
            uint32_t c = H[x];
            value_t val = (value_t)((uint32_t)x + mnu);
            preencher(v + pos, val, c);
            pos += c;
        }
    }
    return 0;
}

static int ordenar_radix(value_t *v, value_t *aux, size_t n, Workspace *ws) {
    if (n < 2) return 0;
    if (n >= 0xFFFFFFFFu) { mergesort_rec(v, aux, 0, n); return 0; }

    int T = threads_para(n);
    value_t mn, mx;
    minmax_par(v, n, T, &mn, &mx);

    uint32_t mnu = (uint32_t)mn;
    uint32_t maxkey = (uint32_t)mx - mnu;
    if (maxkey == 0) return 0;

    int bits = bits_de(maxkey);
    int passes = (bits + 10) / 11;
    int bpp = (bits + passes - 1) / passes;
    size_t radix = (size_t)1 << bpp;
    uint32_t mask = (uint32_t)(radix - 1);

    uint32_t *C = ws_hist(ws, (size_t)T * radix);
    if (!C) { mergesort_rec(v, aux, 0, n); return 0; }

    int pular = 0;
    int resultado_em_aux = 0;

#pragma omp parallel num_threads(T)
    {
        int t = omp_get_thread_num(), nt = omp_get_num_threads();
        size_t a = particao(n, t, nt), b = particao(n, t + 1, nt);
        value_t *src = v, *dst = aux;
        uint32_t *c = C + (size_t)t * radix;

        for (int p = 0; p < passes; ++p) {
            int shift = p * bpp;
            memset(c, 0, radix * sizeof(uint32_t));
            for (size_t i = a; i < b; ++i)
                c[(((uint32_t)src[i] - mnu) >> shift) & mask]++;
#pragma omp barrier
#pragma omp single
            {
                uint32_t corrente = 0;
                int trivial = 0;
                for (size_t d = 0; d < radix; ++d) {
                    uint32_t total = 0;
                    for (int k = 0; k < nt; ++k) {
                        uint32_t q = C[(size_t)k * radix + d];
                        C[(size_t)k * radix + d] = corrente + total;
                        total += q;
                    }
                    if (total == (uint32_t)n) trivial = 1;
                    corrente += total;
                }
                pular = trivial;
            }
            if (!pular) {
                for (size_t i = a; i < b; ++i) {
                    value_t x = src[i];
                    dst[c[(((uint32_t)x - mnu) >> shift) & mask]++] = x;
                }
                value_t *tmp = src; src = dst; dst = tmp;
            }
#pragma omp barrier
        }
        if (t == 0) resultado_em_aux = (src == aux);
    }

    if (resultado_em_aux) copia_paralela(v, aux, n, T);
    return 0;
}

static size_t corank(size_t m, const value_t *A, size_t na, const value_t *B, size_t nb) {
    size_t lo = (m > nb) ? m - nb : 0;
    size_t hi = (m < na) ? m : na;
    while (lo < hi) {
        size_t i = lo + (hi - lo) / 2;
        size_t j = m - i;
        if (A[i] <= B[j - 1]) lo = i + 1;
        else hi = i;
    }
    return lo;
}

static void merge_seq(const value_t *A, size_t na, const value_t *B, size_t nb, value_t *out) {
    size_t i = 0, j = 0, k = 0;
    while (i < na && j < nb) {
        value_t a = A[i], b = B[j];
        int pega_a = (a <= b);
        out[k++] = pega_a ? a : b;
        i += pega_a;
        j += !pega_a;
    }
    if (i < na) memcpy(out + k, A + i, (na - i) * sizeof(value_t));
    else if (j < nb) memcpy(out + k, B + j, (nb - j) * sizeof(value_t));
}

static void ordenar_merge_paralelo(value_t *v, value_t *aux, size_t n) {
    int T = threads_para(n);
    if (T == 1 || n < ((size_t)1 << 16)) { mergesort_rec(v, aux, 0, n); return; }

    int C = 1;
    while (C < T) C <<= 1;
    if (C > MAX_T) C = MAX_T;

    size_t lim[MAX_T + 1];
    for (int i = 0; i <= C; ++i) lim[i] = particao(n, i, C);

#pragma omp parallel for num_threads(T) schedule(dynamic, 1)
    for (int i = 0; i < C; ++i)
        mergesort_rec(v, aux, lim[i], lim[i + 1]);

    value_t *src = v, *dst = aux;
    for (int w = 1; w < C; w *= 2) {
        int pares = C / (2 * w);
        int partes = (T + pares - 1) / pares;
        int tarefas = pares * partes;
        const value_t *S = src;
        value_t *D = dst;

#pragma omp parallel for num_threads(T) schedule(static)
        for (int tk = 0; tk < tarefas; ++tk) {
            int par = tk / partes, parte = tk % partes;
            int ci = par * 2 * w;
            size_t s0 = lim[ci], sm = lim[ci + w], s1 = lim[ci + 2 * w];
            const value_t *A = S + s0, *B = S + sm;
            size_t na = sm - s0, nb = s1 - sm, L = na + nb;
            size_t m0 = (size_t)((unsigned long long)L * parte / partes);
            size_t m1 = (size_t)((unsigned long long)L * (parte + 1) / partes);
            size_t i0 = corank(m0, A, na, B, nb), i1 = corank(m1, A, na, B, nb);
            size_t j0 = m0 - i0, j1 = m1 - i1;
            merge_seq(A + i0, i1 - i0, B + j0, j1 - j0, D + s0 + m0);
        }
        value_t *tmp = src; src = dst; dst = tmp;
    }
    if (src != v) copia_paralela(v, src, n, T);
}

/* ------------------------------------------------------------------ */
/* Orquestração e Execução                                            */
/* ------------------------------------------------------------------ */

enum { A_CONTAGEM, A_RADIX, A_MERGE_PAR, A_INTRO_PAR, A_INTRO, A_MERGE, A_HEAP, A_QSORT, A_TOTAL };

static const char *NOMES[A_TOTAL] = {
    "Counting Sort (paralelo)",
    "Radix Sort LSD (paralelo)",
    "Merge Sort (paralelo)",
    "Intro Sort (paralelo)",
    "Intro Sort (sequencial)",
    "Merge Sort (sequencial)",
    "Heap Sort",
    "qsort (biblioteca C)"
};

static int rodar(int tipo, value_t *v, value_t *aux, size_t n, Workspace *ws) {
    switch (tipo) {
        case A_CONTAGEM:  return ordenar_contagem(v, n, ws);
        case A_RADIX:     return ordenar_radix(v, aux, n, ws);
        case A_MERGE_PAR: ordenar_merge_paralelo(v, aux, n); return 0;
        case A_INTRO_PAR: introsort_paralelo(v, n); return 0;
        case A_INTRO:     introsort(v, n); return 0;
        case A_MERGE:     mergesort_rec(v, aux, 0, n); return 0;
        case A_HEAP:      heapsort_intervalo(v, 0, n); return 0;
        default:          qsort(v, n, sizeof(*v), comparar_int); return 0;
    }
}

static void executar_teste(Resultado *r, int tipo, const value_t *base, size_t n,
                           value_t *copia, value_t *aux, Workspace *ws, Assinatura ref) {
    int reps = (tipo <= A_INTRO_PAR) ? REPETICOES : 1;
    int T = threads_para(n);
    double melhor_clock = 1e300, melhor_parede = 1e300;

    r->nome = NOMES[tipo];
    r->aplicavel = 1;
    r->correto = 1;

    if (reps > 1) {
        copia_paralela(copia, base, n, T);
        if (rodar(tipo, copia, aux, n, ws) != 0) { r->aplicavel = 0; return; }
    }

    for (int k = 0; k < reps; ++k) {
        copia_paralela(copia, base, n, T);

        // AQUICIMENTO DE CPU E THREADS (Lock de Frequencia Turbo Boost)
        aquecer_cpu_e_threads();

        clock_t inicio = clock();
        double w0 = agora_ms();
        int rc = rodar(tipo, copia, aux, n, ws);
        double w1 = agora_ms();
        clock_t fim = clock();

        if (rc != 0) { r->aplicavel = 0; return; }

        double ms_clock = 1000.0 * (double)(fim - inicio) / CLOCKS_PER_SEC;
        double ms_parede = w1 - w0;
        if (ms_clock < melhor_clock) melhor_clock = ms_clock;
        if (ms_parede < melhor_parede) melhor_parede = ms_parede;

        Assinatura s = assinatura(copia, n);
        if (!esta_ordenado(copia, n) || s.soma != ref.soma || s.soma2 != ref.soma2)
            r->correto = 0;
    }
    r->clock_ms = melhor_clock;
    r->parede_ms = melhor_parede;
}

int main(int argc, char **argv) {
    const char *caminho = "listaOrdenada.txt";
    for (int i = 1; i < argc; ++i) {
        caminho = argv[i];
    }

    FILE *f_test = fopen(caminho, "r");
    if (!f_test) {
        caminho = "ListahAhSerOrdenada.txt";
        f_test = fopen(caminho, "r");
    }
    if (!f_test) {
        caminho = "listaOdenada.txt";
        f_test = fopen(caminho, "r");
    }
    if (f_test) fclose(f_test);

    if (!getenv("OMP_NUM_THREADS")) {
        omp_set_dynamic(0);
        omp_set_num_threads(omp_get_num_procs());
    }

    size_t n = 0;
    value_t *dados = carregar_arquivo(caminho, &n);

    if (!dados || n == 0) {
        fprintf(stderr, "Erro: nao foi possivel abrir/ler '%s'.\n", caminho);
        return 1;
    }

    Assinatura ref = assinatura(dados, n);
    value_t *copia = (value_t *)malloc_alinhado(n * sizeof(*copia));
    value_t *aux = (value_t *)malloc_alinhado(n * sizeof(*aux));
    if (!copia || !aux) {
        fprintf(stderr, "Memoria insuficiente.\n");
        free_alinhado(copia); free_alinhado(aux); free_alinhado(dados);
        return 1;
    }

    Workspace ws = {NULL, 0};
    Resultado res[A_TOTAL];
    int melhor = -1;

    aquecer_cpu_e_threads();

    printf("\n========================================================================================\n");
#if defined(USA_AVX512)
    printf(" Arquitetura: x86-64 / AVX-512F (+ caminhos AVX2)\n");
#elif defined(USA_AVX2)
    printf(" Arquitetura: x86-64 / AVX2\n");
#elif defined(USA_NEON)
    printf(" Arquitetura: ARM64 / NEON\n");
#elif defined(__x86_64__) || defined(_M_X64)
    printf(" Arquitetura: x86-64 / SIMD generico\n");
#elif defined(__aarch64__) || defined(_M_ARM64)
    printf(" Arquitetura: ARM64 / SIMD generico\n");
#else
    printf(" Arquitetura: generica\n");
#endif
    printf(" BENCHMARK EXTREMO DE ORDENAÇÃO (Elementos: %zu | Threads: %d)\n", n, omp_get_max_threads());
    imprimir_capacidades_cpu();
    printf("========================================================================================\n\n");

    printf("+---------------------------+-----------+------------+-------------+---------+\n");
    printf("| Algoritmo                 | clock (s) | clock (ms) |  Tempo (ms) | status  |\n");
    printf("+---------------------------+-----------+------------+-------------+---------+\n");

    for (int i = 0; i < A_TOTAL; ++i) {
        executar_teste(&res[i], i, dados, n, copia, aux, &ws, ref);
        if (!res[i].aplicavel) {
            printf("| %-25s | %9s | %10s | %11s | N/A     |\n", res[i].nome, "-", "-", "-");
            continue;
        }
        printf("| %-25s | %9.4f | %10.2f | %11.2f | %-7s |\n",
               res[i].nome, res[i].clock_ms / 1000.0, res[i].clock_ms, res[i].parede_ms,
               res[i].correto ? "SUCESSO" : "ERRO");
        fflush(stdout);
        if (melhor < 0 || res[i].parede_ms < res[melhor].parede_ms) melhor = i;
    }

    printf("+---------------------------+-----------+------------+-------------+---------+\n\n");

    if (melhor >= 0)
        printf("Vencedor Absoluto: %s (%.2f ms de tempo real)\n\n", res[melhor].nome, res[melhor].parede_ms);

    free_alinhado(copia); free_alinhado(aux); free_alinhado(ws.hist); free_alinhado(dados);
    return 0;
}
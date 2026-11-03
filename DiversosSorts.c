/*
 * Benchmark de ordenacao - versao paralela (OpenMP) e otimizada.
 *
 * Alvo x86-64-v3: AVX2 + BMI1/BMI2 + FMA + LZCNT + MOVBE + F16C (Intel Haswell /
 * AMD Excavator/Zen ou mais novos). Os kernels AVX2 sao compilados quando o
 * compilador define __AVX2__; sem isso o programa usa as versoes escalares.
 *
 * Compilar (GCC 11+ / Clang 12+ / MinGW-w64 / Linux):
 *   gcc -O3 -march=x86-64-v3 -funroll-loops -flto -fopenmp DiversosSorts_1.c -o DiversosSorts
 *   (GCC mais antigo: -mavx2 -mbmi -mbmi2 -mlzcnt -mfma -mmovbe -mf16c)
 *   No GCC, mesmo sem -march o proprio codigo liga o conjunto v3 (pragma);
 *   para desligar isso e usar so codigo escalar: -DSEM_V3
 *
 * Compilar (MSVC, Developer Command Prompt):
 *   cl /O2 /openmp /arch:AVX2 DiversosSorts_1.c
 *
 * Executar (basta rodar: ja testa TODOS os algoritmos, sem precisar de flag):
 *   DiversosSorts                      (usa listaOdenada.txt)
 *   DiversosSorts outroArquivo.txt
 *   DiversosSorts outroArquivo.txt --competicao   (so o algoritmo mais adequado)
 *
 * Cada ordenacao e cronometrada com clock() imediatamente antes e depois da
 * chamada; a leitura do arquivo e a verificacao de corretude ficam fora da
 * medicao. Alem do clock() mostramos o tempo de parede (omp_get_wtime), que e o
 * tempo "real" quando ha varias threads (no Linux/macOS clock() soma o tempo de
 * CPU de todas as threads; no Windows ele e tempo de parede).
 *
 * Algoritmos:
 *   1) Counting Sort paralelo   - O(n + faixa), ideal quando os valores sao "pequenos"
 *   2) Radix Sort LSD paralelo  - O(n * passes), funciona para qualquer int de 32 bits
 *   3) Merge Sort paralelo      - Merge Sort por bloco + merge em arvore com
 *                                 particionamento "merge path"
 *   4) Intro Sort paralelo      - particiona em paralelo (nivel a nivel) e ordena
 *                                 os pedacos com Intro Sort, um por thread
 *   Para comparacao: Intro Sort, Merge Sort sequencial, Heap Sort e qsort da
 *   biblioteca C (estes sao sequenciais por natureza, usam 1 nucleo).
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <limits.h>

#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_thread_num(void)  { return 0; }
static int omp_get_num_threads(void) { return 1; }
static int omp_get_max_threads(void) { return 1; }
static int omp_get_num_procs(void)   { return 1; }
static void omp_set_num_threads(int n) { (void)n; }
static void omp_set_dynamic(int d)     { (void)d; }
#endif

/* ------------------------------------------------------------------ */
/* x86-64-v3 (AVX2, BMI2, LZCNT...)                                    */
/* ------------------------------------------------------------------ */
#if (defined(__x86_64__) || defined(_M_X64)) && !defined(SEM_V3)
#  if defined(__GNUC__) && !defined(__clang__) && !defined(__AVX2__)
#    pragma GCC target("avx2,bmi,bmi2,lzcnt,fma,movbe,f16c")
#  endif
#  if defined(__AVX2__)
#    include <immintrin.h>
#    define USA_AVX2 1
#  endif
#endif

typedef int value_t;

#define MAX_T 256
#define MAX_SEG 1024 /* maximo de pedacos do Intro Sort paralelo */
#define ORCAMENTO_HIST ((size_t)512 << 20) /* limite de memoria p/ histogramas do Counting Sort */
#define REPETICOES 7                       /* melhor de N execucoes */

typedef struct {
    const char *nome;
    double clock_ms;   /* melhor tempo medido com clock()            */
    double parede_ms;  /* melhor tempo de parede (omp_get_wtime)     */
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
/* Utilidades                                                          */
/* ------------------------------------------------------------------ */

static double agora_ms(void) {
#ifdef _OPENMP
    return omp_get_wtime() * 1000.0;
#else
    return 1000.0 * (double)clock() / CLOCKS_PER_SEC;
#endif
}

/* Numero de bits necessarios para x (0 -> 0). Vira LZCNT em x86-64-v3. */
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
        free(ws->hist);
        ws->hist = (uint32_t *)malloc(elementos * sizeof(uint32_t));
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
    Assinatura a = {s1, s2};
    return a;
}

/* Confere os pares (i-1, i) para i em [a, b). */
static int bloco_ordenado(const value_t *v, size_t a, size_t b) {
    size_t i = a;
#ifdef USA_AVX2
    for (; i + 8 <= b; i += 8) {
        __m256i x = _mm256_loadu_si256((const __m256i *)(v + i - 1));
        __m256i y = _mm256_loadu_si256((const __m256i *)(v + i));
        if (_mm256_movemask_epi8(_mm256_cmpgt_epi32(x, y))) return 0;
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

/* Cria as threads do OpenMP antes da primeira medicao. */
static void aquecer_threads(void) {
    volatile int sink = 0;
#pragma omp parallel
    {
#pragma omp atomic
        sink += 1;
    }
    (void)sink;
}

static void copia_paralela(value_t *dst, const value_t *src, size_t n, int T) {
#pragma omp parallel num_threads(T)
    {
        int t = omp_get_thread_num(), nt = omp_get_num_threads();
        size_t a = particao(n, t, nt), b = particao(n, t + 1, nt);
        memcpy(dst + a, src + a, (b - a) * sizeof(value_t));
    }
}

/* Minimo e maximo de v[a, b). AVX2: 4 acumuladores x 8 inteiros por iteracao. */
static void minmax_bloco(const value_t *v, size_t a, size_t b, value_t *lo_out, value_t *hi_out) {
    value_t lo = INT_MAX, hi = INT_MIN;
    size_t i = a;
#ifdef USA_AVX2
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

/* ------------------------------------------------------------------ */
/* Leitura paralela do arquivo                                         */
/* ------------------------------------------------------------------ */

#define EH_TOKEN(c) ((((unsigned)(c) - '0') < 10u) || (c) == '-')

/* Cada thread processa os numeros cujo inicio cai em [lo, hi).
 * Com out == NULL apenas conta. */
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

#ifdef _WIN32
    _fseeki64(f, 0, SEEK_END);
    long long tam = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
#else
    fseek(f, 0, SEEK_END);
    long long tam = ftell(f);
    fseek(f, 0, SEEK_SET);
#endif
    if (tam < 0) { fclose(f); return NULL; }

    unsigned char *buf = (unsigned char *)malloc((size_t)tam + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t lidos = fread(buf, 1, (size_t)tam, f);
    fclose(f);
    buf[lidos] = 0;

    int T = ((size_t)lidos < ((size_t)1 << 20)) ? 1 : omp_get_max_threads();
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
            dados = (value_t *)malloc((total ? total : 1) * sizeof(value_t));
        }
        if (dados) varrer(buf, lo, hi, dados + off[t]);
    }

    free(buf);
    *n_out = dados ? total : 0;
    return dados;
}

/* ------------------------------------------------------------------ */
/* Merge Sort sequencial (usado como base do Merge Sort paralelo)      */
/* ------------------------------------------------------------------ */

static void insertion_sort(value_t *v, size_t inicio, size_t fim);
static void merge_seq(const value_t *A, size_t na, const value_t *B, size_t nb, value_t *out);

#ifdef USA_AVX2
#define CORTE_INS 64 /* folhas de 64 inteiros ordenadas com rede AVX2 */

#define CE(a, b) do { __m256i t_ = _mm256_min_epi32(a, b); b = _mm256_max_epi32(a, b); a = t_; } while (0)

/* Transposicao 8x8 de inteiros de 32 bits. */
static inline void transpor8x8(__m256i *r) {
    __m256i t0 = _mm256_unpacklo_epi32(r[0], r[1]), t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    __m256i t2 = _mm256_unpacklo_epi32(r[2], r[3]), t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    __m256i t4 = _mm256_unpacklo_epi32(r[4], r[5]), t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    __m256i t6 = _mm256_unpacklo_epi32(r[6], r[7]), t7 = _mm256_unpackhi_epi32(r[6], r[7]);
    __m256i u0 = _mm256_unpacklo_epi64(t0, t2), u1 = _mm256_unpackhi_epi64(t0, t2);
    __m256i u2 = _mm256_unpacklo_epi64(t1, t3), u3 = _mm256_unpackhi_epi64(t1, t3);
    __m256i u4 = _mm256_unpacklo_epi64(t4, t6), u5 = _mm256_unpackhi_epi64(t4, t6);
    __m256i u6 = _mm256_unpacklo_epi64(t5, t7), u7 = _mm256_unpackhi_epi64(t5, t7);
    r[0] = _mm256_permute2x128_si256(u0, u4, 0x20);
    r[1] = _mm256_permute2x128_si256(u1, u5, 0x20);
    r[2] = _mm256_permute2x128_si256(u2, u6, 0x20);
    r[3] = _mm256_permute2x128_si256(u3, u7, 0x20);
    r[4] = _mm256_permute2x128_si256(u0, u4, 0x31);
    r[5] = _mm256_permute2x128_si256(u1, u5, 0x31);
    r[6] = _mm256_permute2x128_si256(u2, u6, 0x31);
    r[7] = _mm256_permute2x128_si256(u3, u7, 0x31);
}

/* Ordena 64 inteiros em p (scratch: outros 64 inteiros, sobrescritos).
 * 1) rede de 19 comparadores ordena as 8 colunas ao mesmo tempo (min/max AVX2);
 * 2) transpoe: viram 8 corridas ordenadas de 8; 3) 3 merges escalares sem desvio. */
static void sort64_avx2(value_t *p, value_t *scratch) {
    __m256i r[8];
    for (int i = 0; i < 8; ++i) r[i] = _mm256_loadu_si256((const __m256i *)(p + 8 * i));
    CE(r[0], r[2]); CE(r[1], r[3]); CE(r[4], r[6]); CE(r[5], r[7]);
    CE(r[0], r[4]); CE(r[1], r[5]); CE(r[2], r[6]); CE(r[3], r[7]);
    CE(r[0], r[1]); CE(r[2], r[3]); CE(r[4], r[5]); CE(r[6], r[7]);
    CE(r[2], r[4]); CE(r[3], r[5]);
    CE(r[1], r[4]); CE(r[3], r[6]);
    CE(r[1], r[2]); CE(r[3], r[4]); CE(r[5], r[6]);
    transpor8x8(r);
    for (int i = 0; i < 8; ++i) _mm256_storeu_si256((__m256i *)(scratch + 8 * i), r[i]);
    for (int k = 0; k < 4; ++k) merge_seq(scratch + 16 * k, 8, scratch + 16 * k + 8, 8, p + 16 * k);
    for (int k = 0; k < 2; ++k) merge_seq(p + 32 * k, 16, p + 32 * k + 16, 16, scratch + 32 * k);
    merge_seq(scratch, 32, scratch + 32, 32, p);
}
#else
#define CORTE_INS 32 /* abaixo disso o Insertion Sort e mais rapido que dividir */
#endif

/* Merge Sort "ping-pong": src e dst tem o mesmo conteudo em [ini, fim) na
 * entrada; ao final a faixa ordenada esta em dst. Alternar os papeis dos dois
 * vetores evita copiar de volta a cada nivel da recursao. */
static void msort_pp(value_t *dst, value_t *src, size_t ini, size_t fim) {
    if (fim - ini <= CORTE_INS) {
#ifdef USA_AVX2
        if (fim - ini == 64) { sort64_avx2(dst + ini, src + ini); return; }
#endif
        insertion_sort(dst, ini, fim);
        return;
    }

#ifdef USA_AVX2
    size_t meio = ini + (((fim - ini) / 2 + 32) / 64) * 64;   /* folhas de exatamente 64 */
#else
    size_t meio = ini + (fim - ini) / 2;
#endif
    msort_pp(src, dst, ini, meio);
    msort_pp(src, dst, meio, fim);

    if (src[meio - 1] <= src[meio]) { /* ja em ordem: so copia */
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

/* Ordena v[inicio, fim) usando aux[inicio, fim) como area de trabalho. */
static void mergesort_rec(value_t *v, value_t *aux, size_t inicio, size_t fim) {
    if (fim - inicio < 2) return;
    memcpy(aux + inicio, v + inicio, (fim - inicio) * sizeof(*v));
    msort_pp(v, aux, inicio, fim);
}

/* ------------------------------------------------------------------ */
/* Algoritmos classicos (referencia, usados no modo --todos)           */
/* ------------------------------------------------------------------ */

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
        value_t temp = v[raiz];
        v[raiz] = v[filho];
        v[filho] = temp;
        raiz = filho;
    }
}

static void heapsort_intervalo(value_t *v, size_t inicio, size_t fim) {
    size_t n = fim - inicio;
    if (n < 2) return;
    for (size_t i = n / 2; i > 0; --i)
        sift_down(v + inicio, i - 1, n);
    for (size_t fim_heap = n; fim_heap > 1; --fim_heap) {
        value_t temp = v[inicio];
        v[inicio] = v[inicio + fim_heap - 1];
        v[inicio + fim_heap - 1] = temp;
        sift_down(v + inicio, 0, fim_heap - 1);
    }
}

/* Particiona v[inicio, fim) (fim - inicio > 24) com mediana de 3 e devolve o
 * ponto de corte: tudo em [inicio, corte) <= tudo em [corte, fim). */
static size_t particionar(value_t *v, size_t inicio, size_t fim) {
    size_t meio = inicio + (fim - inicio) / 2;
    size_t ultimo = fim - 1;
    if (v[meio] < v[inicio]) { value_t t=v[meio]; v[meio]=v[inicio]; v[inicio]=t; }
    if (v[ultimo] < v[meio]) { value_t t=v[ultimo]; v[ultimo]=v[meio]; v[meio]=t; }
    if (v[meio] < v[inicio]) { value_t t=v[meio]; v[meio]=v[inicio]; v[inicio]=t; }
    value_t pivo = v[meio];

    size_t i = inicio, j = fim - 1;
    for (;;) {
        while (v[i] < pivo) ++i;
        while (v[j] > pivo) --j;
        if (i >= j) break;
        value_t t = v[i]; v[i] = v[j]; v[j] = t;
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
    if (fim > inicio) insertion_sort(v, inicio, fim);
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

/* Intro Sort paralelo (compativel com OpenMP 2.0, sem "tasks"):
 *  1. particiona nivel a nivel; em cada nivel todos os pedacos sao
 *     particionados em paralelo (o 1o nivel usa 1 thread, o 2o usa 2, ...);
 *  2. ao ter ~4 pedacos por thread, cada pedaco e ordenado por uma thread com
 *     o Intro Sort sequencial (escalonamento dinamico equilibra a carga). */
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

/* ------------------------------------------------------------------ */
/* 1: Counting Sort paralelo                                           */
/* ------------------------------------------------------------------ */
/* Fases (tudo em uma unica regiao paralela):
 *  1. cada thread zera e preenche o proprio histograma do seu bloco;
 *  2. cada thread soma os histogramas numa fatia da faixa de valores;
 *  3. prefixo sequencial minusculo entre as fatias;
 *  4. cada thread escreve sua fatia ja ordenada no vetor final.
 * Retorna -1 se a faixa de valores for grande demais (use o Radix). */
/* dst[i] += src[i] (somar histogramas das threads) */
static void hist_somar(uint32_t *dst, const uint32_t *src, size_t n) {
    size_t i = 0;
#ifdef USA_AVX2
    for (; i + 8 <= n; i += 8) {
        __m256i a = _mm256_loadu_si256((const __m256i *)(dst + i));
        __m256i b = _mm256_loadu_si256((const __m256i *)(src + i));
        _mm256_storeu_si256((__m256i *)(dst + i), _mm256_add_epi32(a, b));
    }
#endif
    for (; i < n; ++i) dst[i] += src[i];
}

/* Soma dos n contadores (acumula em 64 bits). */
static size_t hist_total(const uint32_t *h, size_t n) {
    size_t i = 0, s = 0;
#ifdef USA_AVX2
    __m256i acc = _mm256_setzero_si256();
    for (; i + 8 <= n; i += 8) {
        __m256i x = _mm256_loadu_si256((const __m256i *)(h + i));
        acc = _mm256_add_epi64(acc, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(x)));
        acc = _mm256_add_epi64(acc, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(x, 1)));
    }
    uint64_t t[4];
    _mm256_storeu_si256((__m256i *)t, acc);
    s = (size_t)(t[0] + t[1] + t[2] + t[3]);
#endif
    for (; i < n; ++i) s += h[i];
    return s;
}

/* dst[0, cnt) = val. Corridas longas: stores AVX2 de 32 bytes (o ultimo
 * sobrepoe o anterior, sempre dentro de [dst, dst+cnt)). */
static inline void preencher(value_t *dst, value_t val, size_t cnt) {
    size_t j = 0;
#ifdef USA_AVX2
    if (cnt >= 16) {
        __m256i vv = _mm256_set1_epi32(val);
        for (; j + 8 <= cnt; j += 8) _mm256_storeu_si256((__m256i *)(dst + j), vv);
        if (j < cnt) _mm256_storeu_si256((__m256i *)(dst + cnt - 8), vv);
        return;
    }
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

/* ------------------------------------------------------------------ */
/* 2: Radix Sort LSD paralelo                                          */
/* ------------------------------------------------------------------ */
/* Subtrai o minimo (aceita negativos), usa so os bits necessarios,
 * digitos de ate 11 bits (2048 baldes) e pula passes triviais. */
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

/* ------------------------------------------------------------------ */
/* 3: Merge Sort paralelo                                              */
/* ------------------------------------------------------------------ */

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

    /* Fase 1: cada bloco e ordenado com Merge Sort sequencial
     * (as faixas de aux sao disjuntas entre os blocos). */
#pragma omp parallel for num_threads(T) schedule(dynamic, 1)
    for (int i = 0; i < C; ++i)
        mergesort_rec(v, aux, lim[i], lim[i + 1]);

    /* Fase 2: merge em arvore, cada merge dividido entre threads (merge path). */
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
/* Execucao dos testes                                                 */
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

/* Retorna 0 se executou, -1 se nao aplicavel */
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

    if (reps > 1) { /* aquecimento: threads, paginas e workspace */
        copia_paralela(copia, base, n, T);
        if (rodar(tipo, copia, aux, n, ws) != 0) { r->aplicavel = 0; return; }
    }

    for (int k = 0; k < reps; ++k) {
        copia_paralela(copia, base, n, T);   /* fora da medicao */

        /* --- inicio da medicao: somente a ordenacao --- */
        clock_t inicio = clock();
        double w0 = agora_ms();
        int rc = rodar(tipo, copia, aux, n, ws);
        double w1 = agora_ms();
        clock_t fim = clock();
        /* --- fim da medicao --- */

        if (rc != 0) { r->aplicavel = 0; return; }

        double tempo_gasto = ((double)(fim - inicio)) / CLOCKS_PER_SEC;
        double ms_clock = tempo_gasto * 1000.0;
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

/* ------------------------------------------------------------------ */
/* Modo competicao                                                     */
/* ------------------------------------------------------------------ */

/* Counting Sort quando a faixa de valores permite; senao Radix Sort.
 * Devolve o indice do algoritmo usado. */
static int ordenar_oficial(value_t *v, value_t *aux, size_t n, Workspace *ws) {
    if (ordenar_contagem(v, n, ws) == 0) return A_CONTAGEM;
    ordenar_radix(v, aux, n, ws);
    return A_RADIX;
}

static int modo_competicao(value_t *dados, size_t n, Assinatura ref) {
    value_t *aux = (value_t *)malloc(n * sizeof(*aux));
    if (!aux) {
        fprintf(stderr, "Memoria insuficiente.\n");
        return 1;
    }
    Workspace ws = {NULL, 0};

    /* --- inicio da medicao: somente a ordenacao --- */
    clock_t inicio = clock();
    int usado = ordenar_oficial(dados, aux, n, &ws);
    clock_t fim = clock();
    /* --- fim da medicao --- */

    double tempo_gasto = ((double)(fim - inicio)) / CLOCKS_PER_SEC;

    Assinatura s = assinatura(dados, n);
    int ok = esta_ordenado(dados, n) && s.soma == ref.soma && s.soma2 == ref.soma2;

    printf("Elementos: %zu\n", n);
    printf("Algoritmo: %s\n", NOMES[usado]);
    printf("Tempo de ordenacao: %f segundos (%f ms)\n", tempo_gasto, tempo_gasto * 1000.0);
    printf("Verificacao: %s\n", ok ? "OK (ordem crescente, nenhum elemento perdido)" : "FALHA");

    free(aux);
    free(ws.hist);
    return ok ? 0 : 2;
}

/* ------------------------------------------------------------------ */
/* Modo padrao: testa todos os algoritmos e compara os tempos          */
/* ------------------------------------------------------------------ */

static int modo_todos(value_t *dados, size_t n, Assinatura ref) {
    value_t *copia = (value_t *)malloc(n * sizeof(*copia));
    value_t *aux = (value_t *)malloc(n * sizeof(*aux));
    if (!copia || !aux) {
        fprintf(stderr, "Memoria insuficiente para executar os benchmarks.\n");
        free(copia); free(aux);
        return 1;
    }

    Workspace ws = {NULL, 0};
    Resultado res[A_TOTAL];
    int melhor = -1;

    aquecer_threads();

    printf("Elementos: %zu | Threads: %d\n", n, omp_get_max_threads());
    printf("Tempo medido com clock() ao redor de cada ordenacao (leitura e verificacao ficam de fora).\n");
    printf("Algoritmos paralelos: melhor de %d execucoes (apos aquecimento); os demais: 1 execucao.\n\n", REPETICOES);
    printf("%-25s | %12s | %12s | %12s | %s\n", "Algoritmo", "clock (s)", "clock (ms)", "parede (ms)", "status");
    printf("--------------------------+--------------+--------------+--------------+-------\n");

    for (int i = 0; i < A_TOTAL; ++i) {
        executar_teste(&res[i], i, dados, n, copia, aux, &ws, ref);
        if (!res[i].aplicavel) {
            printf("%-25s | %12s | %12s | %12s | nao aplicavel a esta faixa\n", res[i].nome, "-", "-", "-");
            continue;
        }
        printf("%-25s | %12.6f | %12.3f | %12.3f | %s\n", res[i].nome,
               res[i].clock_ms / 1000.0, res[i].clock_ms, res[i].parede_ms,
               res[i].correto ? "OK" : "FALHA");
        fflush(stdout);
        if (!res[i].correto) {
            fprintf(stderr, "ERRO: algoritmo %s nao ordenou corretamente.\n", res[i].nome);
            free(copia); free(aux); free(ws.hist);
            return 2;
        }
        if (melhor < 0 || res[i].parede_ms < res[melhor].parede_ms) melhor = i;
    }

    if (melhor >= 0)
        printf("\nMais rapido nesta execucao: %s (%.3f ms de parede)\n", res[melhor].nome, res[melhor].parede_ms);
    if (melhor >= 0 && res[A_QSORT].aplicavel && res[melhor].parede_ms > 0)
        printf("Speedup sobre qsort: %.1fx\n", res[A_QSORT].parede_ms / res[melhor].parede_ms);
#if defined(_OPENMP) && !defined(_WIN32)
    printf("\nObs.: neste sistema clock() soma o tempo de CPU de todas as threads; para os\n"
           "algoritmos paralelos o tempo real e o da coluna 'parede'.\n");
#endif

    free(copia); free(aux); free(ws.hist);
    return 0;
}

int main(int argc, char **argv) {
    const char *caminho = "listaOdenada.txt";
    int competicao = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--competicao") == 0) competicao = 1;
        else if (strcmp(argv[i], "--todos") == 0) competicao = 0; /* ja e o padrao */
        else caminho = argv[i];
    }

    /* Usa todos os nucleos logicos (a menos que OMP_NUM_THREADS tenha sido definido). */
    if (!getenv("OMP_NUM_THREADS")) {
        omp_set_dynamic(0);
        omp_set_num_threads(omp_get_num_procs());
    }

    size_t n = 0;
    value_t *dados = carregar_arquivo(caminho, &n);   /* leitura: fora da medicao */

    if (!dados) {
        fprintf(stderr, "Erro: nao foi possivel abrir/ler '%s'.\n", caminho);
        fprintf(stderr, "Uso: %s [arquivo.txt] [--competicao]\n", argv[0]);
        return 1;
    }
    if (n == 0) {
        fprintf(stderr, "Erro: nenhum numero inteiro encontrado no arquivo.\n");
        free(dados);
        return 1;
    }

    Assinatura ref = assinatura(dados, n);
    int rc = competicao ? modo_competicao(dados, n, ref) : modo_todos(dados, n, ref);

    free(dados);
    return rc;
}

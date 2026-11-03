# Competição de Ordenação — O Desafio dos Milissegundos

Programa em C altamente otimizado que lê uma lista grande de números inteiros de um arquivo de texto e executa um benchmark completo entre diversos algoritmos de ordenação, destacando o Counting Sort e Radix Sort paralelos implementados com OpenMP.

No modo padrão (benchmark completo), o programa testa todos os algoritmos suportados, realiza execuções de aquecimento (warm-up), mede o tempo de CPU (`clock()`) e de parede (`omp_get_wtime()`) de cada um, exibe a quantidade de elementos, o número de threads utilizadas e a comparação de desempenho. No modo competição (`--competicao`), executa o algoritmo mais adequado para a faixa de valores informada.

Ao final, o programa valida silenciosamente a ordenação e a integridade dos dados (soma e soma dos quadrados dos elementos).

## Como Funciona

### Algoritmos Testados

1. Counting Sort (paralelo): Algoritmo de contagem $O(n + \text{faixa})$, extremamente rápido para listas com amplitude pequena de valores (ex.: de 1 a 2.500.000).

2. Radix Sort LSD (paralelo): Algoritmo por dígitos $O(n \times \text{passes})$, que lida com qualquer inteiro de 32 bits em sub-passes paralelos.

3. Merge Sort (paralelo): Divisão em blocos por thread e fusão em árvore utilizando particionamento *merge path*.

4. Intro Sort (paralelo): Particionamento paralelo em múltiplos níveis com execução final de Intro Sort em cada bloco.

5. Algoritmos Sequenciais de Referência: Intro Sort sequencial, Merge Sort sequencial, Heap Sort e `qsort` da biblioteca padrão C.

### Fases do Counting Sort Paralelo

1. Contagem Local: Cada thread conta os valores do seu bloco da lista em um histograma próprio.

2. Redução: Cada thread soma os histogramas de todas as threads em uma fatia da faixa de valores.

3. Cálculo de Prefixo: Um cálculo sequencial reduzido define o deslocamento de cada fatia na lista final.

4. Escrita Paralela: Cada thread grava sua fatia já ordenada diretamente no vetor de destino.

### Detalhes de Implementação e Otimização

* Suporte AVX2 / x86-64-v3: Utiliza instruções vetoriais AVX2 quando suportadas pelo compilador/processador para min/max, somas e preenchimento de memória.

* Leitura Paralela de Arquivo: Leitura e conversão rápida de inteiros dividida em blocos entre as threads.

* Suporte a Números Negativos: O programa calcula o menor valor da lista e aplica o deslocamento necessário antes da contagem ou ordenação por dígitos.

* Medição de Tempo Precisa: Medição isolada estritamente na etapa de ordenação. Utiliza `omp_get_wtime()` para tempo de parede (tempo real decorrido) e `clock()` para tempo de CPU.

* Aquecimento (Warm-up): Realiza execuções prévias para aquecimento de threads, tabelas de página e cache de memória antes de registrar a medição oficial (retendo o menor tempo entre 7 repetições nos algoritmos paralelos).

* Validação de Integridade: Confere se o resultado final está em ordem crescente e se a soma total e a soma dos quadrados batem com os valores originais.

## Medição de Tempo e Precisão

Para garantir medições de milissegundos precisas e justas entre abordagens paralelas e sequenciais:

* Relógio de Parede (`omp_get_wtime()`): Mede o tempo decorrido real. Em sistemas Linux e macOS, o `clock()` padrão da C stdlib acumula o tempo de CPU de todas as threads ativas (ex.: 8 threads rodando por 10 ms reportam 80 ms de CPU). O `omp_get_wtime()` corrige essa distorção refletindo o tempo real percebido pelo utilizador.

* Precisão no Windows: O `clock()` no Windows possui resolução limitada (1 ms). O uso de `omp_get_wtime()` garante resolução de alta precisão (sub-milissegundo).

* Execução de Aquecimento (Warm-up): A primeira execução de um algoritmo é descartada para evitar variações causadas pela alocação de memória, agendamento inicial de threads e carregamento de cache.

* Menor Tempo de N Execuções: O programa repete os testes paralelos 7 vezes e reporta o menor tempo medido, eliminando ruídos causados por outros processos do sistema operacional.

## Requisitos

* Compilador GCC com suporte a OpenMP (MinGW-w64 no Windows ou GCC nativo no Linux).

* O compilador `cl` do Visual Studio (MSVC) também é suportado.

Para verificar se o GCC está instalado:

```
gcc --version
```

## Compilação e Execução

### Via Terminal (GCC / Linux ou Windows)

Compilar com otimizações máximas e suporte a OpenMP:

```
gcc -O3 -march=x86-64-v3 -funroll-loops -flto -fopenmp DiversosSorts_1.c -o DiversosSorts
```

Para compiladores mais antigos sem suporte ao target `x86-64-v3`:

```
gcc -O3 -mavx2 -mbmi -mbmi2 -mlzcnt -mfma -fopenmp DiversosSorts_1.c -o DiversosSorts
```

### Rodar o Benchmark Completo (Todos os Algoritmos)

```
# Linux / Bash
./DiversosSorts listaOdenada.txt

# PowerShell / CMD (Windows)
.\DiversosSorts.exe listaOdenada.txt
```

*(Se nenhum arquivo for informado, o programa tenta abrir `listaOdenada.txt` por padrão).*

### Rodar no Modo Competição (Apenas o Algoritmo Ideal)

```
.\DiversosSorts.exe listaOdenada.txt --competicao
```

### Via Visual Studio (Developer Command Prompt)

```
cl /O2 /openmp /arch:AVX2 DiversosSorts_1.c
```

### Via Extensão C/C++ Compile Run no VS Code

Em Ctrl + , (Configurações), configure apenas estas opções:

| Configuração | Valor |
| --- | --- |
| `c-cpp-compile-run.c-flags` | `-O3 -march=native -fopenmp` |
| `c-cpp-compile-run.c-linker-flags` | `-fopenmp` |

Com o arquivo `.c` aberto, pressione F6 para compilar e rodar.

## Exemplo de Saída

### Modo Benchmark Completo (`--todos`)

```
Elementos: 10000000 | Threads: 8
Tempo medido com clock() ao redor de cada ordenacao (leitura e verificacao ficam de fora).
Algoritmos paralelos: melhor de 7 execucoes (apos aquecimento); os demais: 1 execucao.

Algoritmo                 |    clock (s) |   clock (ms) |  parede (ms) | status
--------------------------+--------------+--------------+--------------+-------
Counting Sort (paralelo)  |     0.184000 |      184.000 |       24.120 | OK
Radix Sort LSD (paralelo) |     0.312000 |      312.000 |       41.500 | OK
Merge Sort (paralelo)     |     0.840000 |      840.000 |      112.300 | OK
Intro Sort (paralelo)     |     1.120000 |     1120.000 |      148.900 | OK
Intro Sort (sequencial)   |     0.680000 |      680.000 |      680.100 | OK
Merge Sort (sequencial)   |     0.890000 |      890.000 |      890.400 | OK
Heap Sort                 |     1.750000 |     1750.000 |     1750.200 | OK
qsort (biblioteca C)      |     1.210000 |     1210.000 |     1210.500 | OK

Mais rapido nesta execucao: Counting Sort (paralelo) (24.120 ms de parede)
Speedup sobre qsort: 50.2x
```

### Modo Competição (`--competicao`)

```
Elementos: 10000000
Algoritmo: Counting Sort (paralelo)
Tempo de ordenacao: 0.024120 segundos (24.120 ms)
Verificacao: OK (ordem crescente, nenhum elemento perdido)
```

## Formato do Arquivo de Entrada

Arquivo de texto simples contendo números inteiros separados por espaço ou quebra de linha:

```
5
3
10
1
```

## Limitações

* O Counting Sort aloca memória para os histogramas de cada thread. Se a amplitude de valores (`máximo - mínimo`) exceder o orçamento estipulado (~512 MB), o programa alterna automaticamente para o Radix Sort ou reporta que não é aplicável.

* Projetado para inteiros de 32 bits (`int`).

* Suporta vetores até o limite de memória disponível da máquina (vetores acima de 4 bilhões de elementos exigem adaptações de tipos de 64 bits em todas as fases).

## Problemas Comuns (FAQ)

### Aparece Threads: 1

O programa foi compilado sem a flag de OpenMP. Certifique-se de incluir `-fopenmp` (GCC) ou `/openmp` (MSVC) na compilação.
Verifique se a variável de ambiente `OMP_NUM_THREADS` não está forçada para 1. No PowerShell: `Remove-Item Env:OMP_NUM_THREADS`.

### gcc is not recognized

O GCC não está no PATH do sistema. Instale o MinGW-w64 e adicione o diretório `bin` às variáveis de ambiente.

### Erro: faixa de valores grande demais

Notificação emitida quando o Counting Sort não pode ser aplicado devido ao tamanho da amplitude de dados. O benchmark ignora o Counting Sort e utiliza o Radix Sort nestes casos.

### Erro: nao foi possivel abrir/ler

O caminho ou nome do arquivo de dados informado no argumento não foi encontrado no diretório.

## Estrutura do Projeto

```
.
├── DiversosSorts_1.c       # código-fonte C otimizado (OpenMP / AVX2)
├── listaOdenada.txt        # lista de números para teste
└── README.md               # documentação do projeto
```
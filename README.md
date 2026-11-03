# Competição de Ordenação - O Desafio dos Milissegundos
Programa em C que lê uma lista grande de números inteiros de um arquivo de texto e a ordena o mais rápido possível usando Counting Sort paralelo com OpenMP. No final, mostra o tipo de ordenação, a quantidade de elementos, o número de threads usadas e o tempo gasto em milissegundos.

Como funciona

O Counting Sort não compara elementos entre si. Ele conta quantas vezes cada valor aparece e depois reescreve a lista já em ordem. Isso é muito mais rápido que quicksort ou mergesort quando os valores estão em uma faixa pequena, como na lista de teste (valores de 1 a 2.500.000).

A versão paralela divide o trabalho entre as threads em quatro fases:

Cada thread conta os valores do seu bloco da lista em um histograma próprio.
Cada thread soma os histogramas de todas as threads em uma fatia da faixa de valores.
Um pequeno cálculo de prefixo define onde cada fatia começa na lista final.
Cada thread escreve sua fatia já ordenada no vetor final.

Outros detalhes:

A leitura do arquivo também é paralela.
Aceita números negativos, pois o programa subtrai o menor valor da lista antes de contar.
O tempo é medido com relógio de parede (omp_get_wtime), somente na ordenação.
Antes da medição há uma execução de aquecimento, que não aparece na tela.
Ao final, o programa confere em silêncio se a lista está em ordem crescente e se a soma e a soma dos quadrados batem com as do original. Só escreve algo se der erro.
Requisitos
Compilador GCC com suporte a OpenMP (MinGW-w64, TDM-GCC ou GCC no Linux).
O cl do Visual Studio também funciona, usando /openmp no lugar de -fopenmp.

Para conferir se o GCC está instalado:

gcc --version
Compilar e executar

No terminal, dentro da pasta do projeto:

gcc -O3 -march=native -fopenmp benchmark_ordenacao.c -o benchmark

Executar passando o arquivo de dados:

benchmark listaOdenada.txt

No PowerShell, use .\benchmark.exe listaOdenada.txt. Se nenhum arquivo for informado, o programa tenta abrir listaOdenada.txt na pasta atual.

Com Visual Studio (Developer Command Prompt):

cl /O2 /openmp /arch:AVX2 benchmark_ordenacao.c
Usando a extensão C/C++ Compile Run no VS Code

Sem criar pasta de projeto nem tasks.json, configure em Ctrl+, (Configurações):

Configuração	Valor
c-cpp-compile-run.c-flags	-O3 -march=native -fopenmp
c-cpp-compile-run.c-linker-flags	-fopenmp
c-cpp-compile-run.run-args	listaOdenada.txt (ou o caminho completo)
c-cpp-compile-run.skip-if-compiled	desmarcado

Depois, com o arquivo .c aberto, aperte F6 para compilar e rodar. A opção skip-if-compiled precisa ficar desligada, senão a extensão pode reaproveitar um executável antigo feito sem OpenMP.

Exemplo de saída
Ordenacao: Counting Sort (paralelo)
Elementos: 10000000
Threads:   8
Tempo:     36.164 ms

Os valores acima são só um exemplo. O tempo e o número de threads dependem da sua máquina.

Formato do arquivo de entrada

Arquivo de texto com números inteiros separados por quebra de linha ou espaço. Exemplo:

5
3
10
1
Limitações
O Counting Sort usa um histograma por thread. Se a faixa de valores (maior menos menor) for grande demais, a memória necessária passa de cerca de 512 MB e o programa para com uma mensagem de erro. Com a lista de teste (faixa de 2,5 milhões) isso não acontece.
Funciona com inteiros de 32 bits (int).
Listas com mais de 4 bilhões de elementos não são suportadas.
Problemas comuns

Aparece Threads: 1

O programa foi compilado sem OpenMP ou o executável é antigo.

Compile com -fopenmp.
Apague o .exe antigo e compile de novo.
Desative skip-if-compiled se estiver usando a extensão do VS Code.
Confira se a variável OMP_NUM_THREADS não está definida como 1. No PowerShell: echo $env:OMP_NUM_THREADS. Para limpar: Remove-Item Env:OMP_NUM_THREADS.

gcc is not recognized

O GCC não está no PATH do Windows. Instale o MinGW-w64 (por exemplo, via MSYS2), adicione a pasta bin ao PATH e reinicie o VS Code.

Erro: faixa de valores grande demais

Os números da lista estão muito espalhados para o Counting Sort. Esse algoritmo só serve para listas com valores em uma faixa limitada.

Erro: nao foi possivel abrir/ler

Confira se o nome e o caminho do arquivo de dados estão corretos.

Estrutura
.
├── benchmark_ordenacao.c   # código-fonte
├── listaOdenada.txt        # lista de números a ordenar
└── README.md

# Competição de Ordenação — O Desafio dos Milissegundos

Programa em C altamente otimizado que lê uma lista grande de números inteiros de um arquivo de texto e a ordena o mais rápido possível usando **Counting Sort paralelo** com **OpenMP**. No final, mostra o tipo de ordenação, a quantidade de elementos, o número de threads usadas e o tempo gasto em milissegundos.

## ⚙️ Como Funciona

O **Counting Sort** não compara elementos entre si. Ele conta quantas vezes cada valor aparece e depois reescreve a lista já em ordem. Isso é muito mais rápido que *Quicksort* ou *Mergesort* quando os valores estão em uma faixa pequena (como na lista de teste, com valores de 1 a 2.500.000).

A versão paralela divide o trabalho entre as threads em quatro fases:

1. **Contagem Local:** Cada thread conta os valores do seu bloco da lista em um histograma próprio.

2. **Redução:** Cada thread soma os histogramas de todas as threads em uma fatia da faixa de valores.

3. **Cálculo de Prefixo:** Um cálculo rápido define onde cada fatia começa na lista final.

4. **Escrita Paralela:** Cada thread escreve sua fatia já ordenada no vetor final.

### Outros detalhes

* **Leitura Otimizada:** A leitura do arquivo também é feita de forma paralela.

* **Suporte a Negativos:** Aceita números negativos, pois o programa subtrai o menor valor da lista antes de contar.

* **Medição Precisa:** O tempo é medido com relógio de parede (`omp_get_wtime`), isolando estritamente a etapa de ordenação.

* **Aquecimento (*Warm-up*):** Antes da medição há uma execução de aquecimento (sem exibir na tela) para estabilizar o cache e o clock do processador.

* **Validação Silenciosa:** Ao final, o programa confere em silêncio se a lista está em ordem crescente e se a soma e a soma dos quadrados batem com as do original. Só exibe mensagem se houver erro.

## ⏱️ Medição de Tempo e Precisão em Milissegundos

Para obter medições confiáveis na casa dos milissegundos, é fundamental utilizar um relógio monotônico de alta resolução (tempo de parede) e seguir boas práticas de benchmark.

### Comportamento dos Relógios por Plataforma

* **Windows (`clock()`):** Possui resolução limitada a cerca de 1 ms (`CLOCKS_PER_SEC` é 1000). Uma ordenação rápida (ex: 0,4 ms) pode ser registrada incorretamente como 0 ms. Para alta precisão nativa no Windows, o recomendado é `QueryPerformanceCounter`.

* **Linux / macOS (`clock()`):** O `clock()` nesses sistemas mede o **tempo total de CPU** somando todas as threads. Em programas paralelos, isso infla o resultado final. Nestas plataformas, utilize `clock_gettime(CLOCK_MONOTONIC)`.

* **OpenMP (`omp_get_wtime()`):** Mede tempo de parede com alta resolução de forma nativa e multiplataforma. É a melhor escolha para benchmarks paralelos.

* **C11 sem OpenMP (`timespec_get`):** Pode ser usado como alternativa portátil, porém não é estritamente monotônico.

### Boas Práticas Adotadas no Benchmark

1. **Rodada de Aquecimento:** Realiza 1 execução prévia descartada para aquecer caches de CPU e estabilizar frequências do processador.

2. **Repetição e Estatística:** Para testes precisos, repita a ordenação de 5 a 10 vezes e considere o **menor tempo** ou a **mediana**, reduzindo interferências do sistema operacional.

3. **Isolamento de Escopo:** Leitura de arquivos, alocação de memória e validações finais ficam totalmente fora da cronometragem.

### Guia para Escolha do Relógio

* **Requisito Acadêmico:** Se o enunciado do trabalho exige `clock()`, mantenha-o como métrica oficial.

* **Comparação Paralela vs. Sequencial:** Use `omp_get_wtime()` por refletir com precisão o ganho de desempenho real (*speedup*).

* **Algoritmos Sequenciais:** O uso de `clock()` só é aceitável caso a execução demore significativamente mais que 1 ms.

## 🛠️ Requisitos

* Compilador **GCC** com suporte a OpenMP (MinGW-w64, TDM-GCC ou GCC no Linux).

* O compilador `cl` do Visual Studio também funciona, utilizando `/openmp` no lugar de `-fopenmp`.

Para conferir se o GCC está instalado:

```
gcc --version

```

## 🚀 Compilar e Executar

### Terminal (GCC)

Para compilar na pasta do projeto:

```
gcc -O3 -march=native -fopenmp benchmark_ordenacao.c -o benchmark

```

Para executar passando o arquivo de dados:

```
# Linux / Bash
./benchmark listaOdenada.txt

# PowerShell (Windows)
.\benchmark.exe listaOdenada.txt

```

*(Se nenhum arquivo for informado, o programa tenta abrir `listaOdenada.txt` na pasta atual).*

### Visual Studio (Developer Command Prompt)

```
cl /O2 /openmp /arch:AVX2 benchmark_ordenacao.c

```

### Extensão *C/C++ Compile Run* no VS Code

Sem criar pasta de projeto ou `tasks.json`, configure em **Ctrl + ,** (Configurações):

| Configuração | Valor | 
| ----- | ----- | 
| `c-cpp-compile-run.c-flags` | `-O3 -march=native -fopenmp` | 
| `c-cpp-compile-run.c-linker-flags` | `-fopenmp` | 

Depois, com o arquivo `.c` aberto, aperte **F6** para compilar e rodar.

## 📊 Exemplo de Saída

```
Ordenacao: Counting Sort (paralelo)
Elementos: 10000000
Threads:   8
Tempo:     36.164 ms

```

*(Valores ilustrativos; tempo e threads variam conforme o hardware).*

## 📄 Formato do Arquivo de Entrada

Arquivo de texto simples com números inteiros separados por quebra de linha ou espaço:

```
5
3
10
1

```

## ⚠️ Limitações

* **Uso de Memória:** O Counting Sort utiliza um histograma por thread. Se a amplitude dos valores (`máximo - mínimo`) for grande demais, a memória necessária pode exceder 512 MB e o programa será encerrado com erro. Na lista de teste padrão (faixa de 2,5 milhões), o consumo é seguro.

* **Tipos de Dados:** Suporta apenas inteiros de 32 bits (`int`).

* **Tamanho da Lista:** Listas com mais de 4 bilhões de elementos não são suportadas.

## 🔍 Problemas Comuns (FAQ)

### Aparece `Threads: 1`

O programa foi compilado sem OpenMP ou executável antigo está sendo executado.

1. Recompile garantindo a flag `-fopenmp`.

2. Apague o executável `.exe` antigo.

3. Verifique a variável `OMP_NUM_THREADS`:

   ```
   # No PowerShell:
   echo $env:OMP_NUM_THREADS
   # Para limpar:
   Remove-Item Env:OMP_NUM_THREADS
   
   ```

### `gcc is not recognized`

O GCC não está no PATH do Windows. Instale o MinGW-w64 (ex: via MSYS2), adicione a pasta `bin` às variáveis de ambiente do sistema e reinicie o editor/terminal.

### `Erro: faixa de valores grande demais`

Os números da lista estão muito espaçados. O Counting Sort exige valores em uma faixa limitada.

### `Erro: nao foi possivel abrir/ler`

Verifique se o nome e o caminho do arquivo de dados estão corretos no diretório de execução.

## 📁 Estrutura do Projeto

```
.
├── benchmark_ordenacao.c   # Código-fonte em C
├── listaOdenada.txt        # Arquivo de dados de entrada
└── README.md               # Documentação do projeto

```
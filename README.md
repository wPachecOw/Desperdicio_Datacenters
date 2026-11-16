# Classificação heurística de desperdício de recursos em cargas de IA

Implementação Python 3.11+ baseada exclusivamente no schema oficial Alibaba GPU v2020.
Leia [DECISOES.md](DECISOES.md) antes de interpretar resultados.

**Não há medição de energia nem identificação de rack.** A saída classifica
subutilização relativa ao pedido da instância. O nome do alvo é
`classe_risco_desperdicio`, mas as classes não são rótulos ambientais validados.

## Executar

O schema é lido da cópia oficial do Alibaba ao lado deste repositório. Se ela
ainda não existir, prepare apenas o subdiretório necessário:

```bash
git clone --filter=blob:none --sparse https://github.com/alibaba/clusterdata.git ../clusterdata
git -C ../clusterdata sparse-checkout set cluster-trace-gpu-v2020
```

A partir desta pasta:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python pipeline.py --download
```

A clonagem existente em `../clusterdata` é reaproveitada: nenhum outro trace é
baixado. Os sete arquivos oficiais v2020 são verificados por SHA-256 e lidos
diretamente de tar.gz em blocos, sem extrair CSVs grandes. Download usa blocos
HTTP Range retomáveis; manter os arquivos `.range-*` e `.partial` permite retomada.
O servidor deve suportar Range. Falha de checksum interrompe a execução.

Amostra padrão: 0,3% de job_ids por hash estável; varre o período completo e
preserva instâncias relacionadas. Não é uma amostra do início do arquivo.
A inspeção informa o tamanho total das tabelas e as estatísticas da seleção.
As integrações são armazenadas em cache para repetir experimentos rapidamente.

```bash
# Alterar fração, janela, pesos e regras: CONFIG no começo de pipeline.py
.venv/bin/python pipeline.py --fracao 0.01 --saida resultados_1pct
.venv/bin/python pipeline.py --split estratificado --saida resultados_estratificados
.venv/bin/python pipeline.py --split servidores --saida resultados_servidores
# JSON substitui as chaves informadas de CONFIG
.venv/bin/python pipeline.py --config exemplo_config.json
# Todas as linhas: exige uma máquina com RAM suficiente para joins/modelagem
.venv/bin/python pipeline.py --fracao 1 --saida resultados_completos
.venv/bin/python -m unittest -v test_pipeline.py
```

Os arquivos compactados somam aproximadamente 1,4 GB. Download retomável mantém
blocos temporários e a montagem do arquivo simultaneamente: reservar pelo menos
3 GB para dados e mais espaço para ambiente/saídas. Leitura em blocos reduz custo
de entrada, mas integração, engenharia e Random Forest usam pandas em memória.
**A execução completa não é out-of-core**: dimensionar RAM pela amostra; não foi
validada em 100% dos dados nesta máquina. Para escala superior à RAM, migrar
joins/agregações para Dask/Spark/DuckDB e manter split e ajuste global no treino.
O modo por servidores pode ser impossível se jobs conectarem o cluster inteiro;
o pipeline falha explicitamente se não houver grupos independentes suficientes.

## Saídas

- `quarentena/*.parquet`: registros originais com identidades/intervalos ou pedidos conflitantes.
- `dicionario_dados.csv`: tipos, papéis e nulos de cada coluna.
- `base_processada.csv` / `.parquet`: dados auditáveis, unidades padronizadas,
  medições e nulos preservados, flags, rótulo e coluna `conjunto`.
- `treino.parquet`, `validacao.parquet`, `teste.parquet`: rastreabilidade por conjunto.
- `*_transformado.npz`: X numérico imputado/codificado/escalonado, y e índices;
  esses arquivos contêm as matrizes prontas para estimadores.
- `preprocessador.joblib`: Pipeline + ColumnTransformer ajustados só no treino.
- `modelo_baseline.joblib`: Random Forest com class_weight=balanced.
- `relatorio_qualidade.md` e `.json`: estatísticas, nulos, joins, classes, seleção,
  outliers, limiares e avaliação do baseline.
- `correlacao_treino.csv`, `importancia_variaveis.csv`, `graficos/*.png`.
- `coortes_inicio.parquet`: resumo descritivo de instâncias pelo início;
  **não** é telemetria temporal nem contagem de jobs ativos em cada janela.

## Modelo e ausência de vazamento

As medições e score constam da base auditável, mas não entram no modelo padrão.
Use somente a lista `preditores_mantidos` do relatório ou o modelo exportado:
passar todas as colunas ao classificador causaria vazamento do alvo heurístico.
O `ColumnTransformer` descarta colunas fora da lista. A inferência pode usar:

```python
import joblib
import pandas as pd
modelo = joblib.load('resultados/modelo_baseline.joblib')
base = pd.read_parquet('resultados/teste.parquet')
# Strings pandas ausentes devem chegar ao sklearn como np.nan.
import numpy as np
pred = modelo.predict(base.astype(object).where(base.notna(), np.nan))
```

Divisão temporal usa timestamps de início distintos e purga intervalos sobrepostos
às fronteiras, além de jobs compartilhados. Estratificação usa grupos de jobs;
servidores usa componentes job-máquina. As proporções são aproximadas.
Desfechos e duração são retrospectivos e excluídos dos preditores. Há diferença
entre prever a heurística e demonstrar desperdício ambiental: a segunda tarefa
exige validação independente com potência, hardware e contexto operacional.

Seleção configurável: `nenhuma`, `kbest`, `forest`, `pca`. Constantes e redundâncias
são identificadas só no treino. PCA mistura categorias codificadas e numerais;
para interpretação, prefira o padrão ou seleção por importância.

Fonte: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020

## Execução realizada com dados reais

Os sete arquivos oficiais foram baixados e passaram nos checksums SHA-256 do
README do Alibaba. A amostra padrão selecionou 22.704 registros de instâncias,
que resultaram em 20.934 identidades auditáveis, com 92 colunas. Destas, 5.903
receberam classe: Baixo 180; Moderado 1.220; Alto 1.715; Crítico 2.788.
Os 15.031 registros sem evidência suficiente foram mantidos sem rótulo.

A divisão temporal produziu 4.024 linhas de treino, 903 de validação e 972 de
teste; quatro linhas rotuladas foram purgadas nas fronteiras. O baseline obteve
balanced accuracy de aproximadamente 0,495 no teste, contra o alvo heurístico.
Esse resultado não valida risco ambiental. Os registros originais com chaves
incompletas ou tentativas conflitantes estão em `resultados/quarentena` (os dois
arquivos podem se sobrepor; não somar seus tamanhos como identidades distintas).

Doze testes automatizados passaram, incluindo exportação, inferência, quarentena,
splits, cache, seletores e ausência de ajuste estatístico com o teste. Os logs da execução local não são versionados. `requirements-lock.txt` registra as versões efetivamente usadas
com Python 3.14.4; para reproduzir o ambiente, instale esse arquivo na venv.

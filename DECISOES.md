# Decisões e schema confirmado antes da implementação

Fonte: ../clusterdata/cluster-trace-gpu-v2020/README.md e data/*.header.
Escolha: v2020 contém medições de utilização; v2023 contém pedidos de recursos,
sem utilização observada. v2025/v2026 não são necessários para este primeiro estudo.

## Tabelas reais (CSV sem cabeçalho, vírgula; leitura UTF-8 estrita)


### pai_group_tag_table

inst_id,user,gpu_type_spec,group,workload

### pai_instance_table

job_name,task_name,inst_name,worker_name,inst_id,status,start_time,end_time,machine

### pai_job_table

job_name,inst_id,user,status,start_time,end_time

### pai_machine_metric

worker_name,machine,start_time,end_time,machine_cpu_iowait,machine_cpu_kernel,machine_cpu_usr,machine_gpu,machine_load_1,machine_net_receive,machine_num_worker,machine_cpu

### pai_machine_spec

machine,gpu_type,cap_cpu,cap_mem,cap_gpu

### pai_sensor_table

job_name,task_name,worker_name,inst_id,machine,gpu_name,cpu_usage,gpu_wrk_util,avg_mem,max_mem,avg_gpu_wrk_mem,max_gpu_wrk_mem,read,write,read_count,write_count

### pai_task_table

job_name,task_name,inst_num,status,start_time,end_time,plan_cpu,plan_mem,plan_gpu,gpu_type

## Unidade e limites

Uma linha representa uma instância de job em uma máquina, durante sua vida observada.
Quando o mesmo identificador tem múltiplos intervalos conflitantes, a identidade
é preservada com intervalo desconhecido, sem classe; os originais ficam em quarentena.
Isso é uma adaptação explícita da unidade solicitada: NÃO existem sensores pontuais
para sustentar job/servidor em janelas. `janela_inicio` agrupa coortes de início,
não mede atividade em cada janela. Médias móveis e variação de uma hora sobre
utilização instantânea, jobs simultâneos, fragmentação física e reinicializações
não são identificáveis com precisão; não serão fabricados. Série temporal mostra
médias de instâncias por início, retrospectivas. `start_time`/`end_time` foram
deslocados: datas são fictícias; hora e semana preservadas em Asia/Shanghai.

Não há rack, energia, potência nominal, temperatura ou refrigeração. Não há kWh.
`workload` é uma etiqueta parcial de modelo/carga, não um tipo completo de job.
Status de falha é desfecho por instância; não contagem de eventos ao longo do tempo.
`servidor_ativo` significa evidência de instância com intervalo válido, não uptime.

## Integração e unidades

Sensores: múltiplas linhas podem representar GPUs de um worker. Antes do join,
exigem-se chaves worker/job/task/inst_id/machine consistentes; múltiplas medições
idênticas são deduplicadas; GPU é somada por dispositivo, CPU/memória de instância
são médias (não somadas por GPU). Sensores repetidos com valores conflitantes geram erro, pois não existe resolução temporal segura.
Instâncias: chave composta worker_name, job_name, task_name, inst_id, machine.
Tentativas com a mesma chave e intervalos diferentes são ambíguas: guardar os
registros originais em quarentena, preservar a identidade com atributos nulos e
flag instancias_ambigua; sem intervalo válido, não há rótulo. Pedidos conflitantes
são tratados da mesma forma, com flag tarefas_ambigua. Tasks: job_name+task_name; tags: inst_id;
spec: machine. Left joins validados many_to_one e auditados. Outras dimensões com chaves conflitantes
interrompem a execução, exigindo investigação. Nulos em chaves são
quarentenados; ausência de dimensão permanece nula. Jobs/machine_metric são
inspecionados; não integrados para evitar duplicação de métricas médias redundantes.

plan_cpu/cpu_usage: dividir por 100 para núcleos; plan_gpu/gpu_wrk_util: dividir
por 100 para GPUs equivalentes. plan_mem/avg_mem: GB. Percentuais usados no score
são frações do recurso solicitado; cap_cpu/cap_mem permitem também fração da
máquina. >100% da alocação é sinalizado e preservado: pode ser uso acima do pedido.
Intensidade usa média de frações adimensionais, nunca soma núcleos+GPUs+GB.

## Ausência, rótulo e vazamento

Rótulo exige pedidos GPU/CPU/memória positivos e medições válidas e duração válida;
sem evidência, classe ausente e linha conservada fora do treinamento. Não imputar
métricas para construir a classe. Score pondera subutilização GPU/CPU/memória;
limiares e pesos são configuráveis. Classes são heurísticas, não ground truth
ambiental. Duração/falha/capacidade são atributos auxiliares, sem impor relação
causal arbitrária com impacto. Não existe validação ambiental independente.

Modelo padrão usa apenas pedidos, tipo de GPU, papel da task e calendário;
exclui score, medições geradoras do alvo, IDs, desfechos e duração retrospectiva.
Mesmo assim, avaliar contra rótulos heurísticos mede sua previsibilidade, não
valida desperdício ambiental. Random Forest com class_weight é baseline; XGBoost,
LightGBM e SVM podem ser comparados após validar a definição do alvo.

Splits: temporal purga instâncias de treino/validação que terminam após a próxima
fronteira, e remove jobs compartilhados; estratificado agrupa por job (aproxima
proporções); opção por servidores usa componentes conectados job-máquina para
não dividir jobs que atravessam servidores. Mediana, imputação categórica,
remoção de constantes, outliers, correlação, seleção e escala ajustados no treino.
Sem interpolação de médias de vida: ela inventaria observações. Picos de IA podem
ser normais: flags não implicam remoção. Identificadores são rastreabilidade.

## Calendário e interpretação das saídas

`indicador_pico` representa uma faixa horária convencional configurável (padrão
09:00–18:00), não um pico de demanda comprovado. `turno` é apenas calendário.
`utilizacao_gpu/cpu/memoria` usa como denominador o recurso solicitado da instância;
as colunas terminadas em `_maquina` usam a capacidade física cadastrada. Nenhuma
dessas razões recupera potência ou calor. GPUs utilizadas são equivalentes de
utilização média, não contagem de placas simultaneamente ativas.

Na base auditável, nulos e valores fora de intervalo permanecem visíveis. Somente
as matrizes `*_transformado.npz` recebem imputação, codificação e escalonamento.
A coluna `motivo_sem_rotulo` explica por que uma identidade ficou fora do treino.
As proporções 70/15/15 são metas: a separação temporal usa períodos e a separação
por grupos usa folds aproximados, priorizando independência.

Os pedidos `plan_cpu`, `plan_gpu` e `plan_mem` são por instância, herdados da task.
Não multiplicamos esses pedidos por `inst_num` ao uni-los às instâncias: isso
contaria o mesmo recurso repetidamente. Nenhuma média de vida é replicada para
preencher janelas temporais sem observações.

As participações na capacidade da máquina são contribuições individuais da
instância, não utilização total do servidor. A tabela `pai_machine_metric`
contém médias globais do servidor durante a vida de cada worker; elas são
inspecionadas, mas não confundidas com uma série temporal independente.

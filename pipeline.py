"""Pré-processamento auditável do Alibaba GPU v2020 (comentários em português).

Execute: .venv/bin/python pipeline.py --download
As medições são médias da vida da instância, não séries temporais instantâneas.
"""
from __future__ import annotations

# Parâmetros centrais; podem ser sobrescritos por JSON via --config.
CONFIG = {
    'fracao_amostra': 0.003, 'seed': 42, 'chunksize': 100_000,
    'janela': '1h', 'pesos': {'gpu': 0.6, 'cpu': 0.2, 'memoria': 0.2},
    'limiares_classes': [0.25, 0.50, 0.75], 'gpu_ociosa_limiar': 10.0,
    'cpu_baixa_limiar': 20.0, 'razao_solicitado_utilizado': 4.0,
    'split': 'temporal', 'proporcoes': [0.70, 0.15, 0.15],
    'selecao': 'nenhuma', 'k_best': 10, 'pca_variancia': 0.95,
    'correlacao_limiar': 0.95, 'iqr_fator': 1.5, 'z_limiar': 3.0,
    'percentis_outliers': [0.01, 0.99], 'pico_horas': [9, 18],
    'treinar_baseline': True,
}

import argparse
import hashlib
import json
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from download import SCHEMA, download_all

ROOT = Path(__file__).resolve().parent
CLASSES = ['Baixo', 'Moderado', 'Alto', 'Crítico']
IDENTIDADE = ['job_name', 'task_name', 'worker_name', 'inst_id', 'machine']
NUMERICAS = {
    'inst_num', 'start_time', 'end_time', 'plan_cpu', 'plan_mem', 'plan_gpu',
    'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem', 'avg_gpu_wrk_mem',
    'max_gpu_wrk_mem', 'read', 'write', 'read_count', 'write_count',
    'cap_cpu', 'cap_mem', 'cap_gpu', 'machine_cpu_iowait', 'machine_cpu_kernel',
    'machine_cpu_usr', 'machine_gpu', 'machine_load_1', 'machine_net_receive',
    'machine_num_worker', 'machine_cpu',
}
# Lista explícita impede IDs, desfechos e componentes do alvo no modelo.
PREDITORES_NUM = ['recursos_cpu_alocados', 'recursos_gpu_alocados', 'memoria_alocada',
                  'hora_do_dia', 'dia_da_semana', 'fim_de_semana', 'indicador_pico']
PREDITORES_CAT = ['task_name', 'gpu_type', 'turno']
LOG = logging.getLogger(__name__)


def salvar_json(path: Path, obj: Any) -> None:
    """Serializa objetos NumPy e timestamps em relatório legível."""
    def normalizar(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): normalizar(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalizar(v) for v in value]
        if isinstance(value, np.generic):
            return normalizar(value.item())
        if value is pd.NA or value is pd.NaT or (isinstance(value, float) and not np.isfinite(value)):
            return None
        return value
    path.write_text(json.dumps(normalizar(obj), ensure_ascii=False, indent=2,
                               default=str, allow_nan=False), encoding='utf-8')


def validar_config(cfg: dict) -> None:
    """Recusa limiares, proporções e opções inconsistentes."""
    if not 0 < cfg['fracao_amostra'] <= 1:
        raise ValueError('fracao_amostra deve estar em (0, 1].')
    p = cfg['proporcoes']
    if len(p) != 3 or min(p) <= 0 or not np.isclose(sum(p), 1):
        raise ValueError('proporcoes deve conter três frações positivas somando 1.')
    limites = cfg['limiares_classes']
    if len(limites) != 3 or not 0 < limites[0] < limites[1] < limites[2] < 1:
        raise ValueError('Três limiares crescentes entre 0 e 1 são necessários.')
    if set(cfg['pesos']) != {'gpu', 'cpu', 'memoria'} or min(cfg['pesos'].values()) <= 0:
        raise ValueError('pesos deve conter gpu, cpu e memoria positivos.')
    if cfg['split'] not in {'temporal', 'estratificado', 'servidores'}:
        raise ValueError('Split desconhecido.')
    if cfg['selecao'] not in {'nenhuma', 'kbest', 'forest', 'pca'}:
        raise ValueError('Seleção desconhecida.')
    if cfg['chunksize'] <= 0 or cfg['k_best'] <= 0:
        raise ValueError('chunksize e k_best devem ser positivos.')
    if not 0 < cfg['pca_variancia'] < 1:
        raise ValueError('pca_variancia deve estar entre 0 e 1.')
    if not 0 <= cfg['percentis_outliers'][0] < cfg['percentis_outliers'][1] <= 1:
        raise ValueError('Percentis de outliers inválidos.')
    pd.tseries.frequencies.to_offset(cfg['janela'])


def iter_csv(data_dir: Path, tabela: str, cfg: dict) -> Iterator[pd.DataFrame]:
    """Lê CSV ou membro .tar.gz em blocos, sem extrair o arquivo completo."""
    cols = (SCHEMA / 'data' / f'{tabela}.header').read_text().strip().split(',')
    options = dict(names=cols, header=None, sep=',', encoding='utf-8',
                   dtype={c: ('float64' if c in NUMERICAS else 'string') for c in cols},
                   chunksize=cfg['chunksize'], on_bad_lines='error')
    csv = data_dir / f'{tabela}.csv'
    if csv.exists():
        yield from pd.read_csv(csv, **options)
    else:
        archive = data_dir / f'{tabela}.tar.gz'
        if not archive.exists():
            raise FileNotFoundError(f'{archive}: execute download.py ou use --download.')
        with tarfile.open(archive, 'r|gz') as tar:
            for member in tar:
                if member.isfile() and Path(member.name).name == f'{tabela}.csv':
                    with tar.extractfile(member) as stream:
                        yield from pd.read_csv(stream, **options)
                    return
        raise ValueError(f'CSV esperado não encontrado em {archive}.')


def amostrar_jobs(ids: pd.Series, cfg: dict) -> pd.Series:
    """Hash estável de job_id seleciona jobs inteiros em todo o período."""
    # Não usa o hash aleatório do Python; não amostra apenas o começo do CSV.
    unique = ids.dropna().unique()
    cutoff = int(cfg['fracao_amostra'] * 2**64)
    chosen = {x for x in unique if int.from_bytes(hashlib.blake2b(
        f"{cfg['seed']}:{x}".encode(), digest_size=8).digest(), 'big') < cutoff}
    return ids.isin(chosen)


def perfil(df: pd.DataFrame) -> dict:
    """Resume a seleção carregada; contagem bruta é reportada separadamente."""
    categories = df.select_dtypes(include=['string', 'object'])
    return {'linhas': len(df), 'colunas': len(df.columns),
            'memoria_bytes': int(df.memory_usage(deep=True).sum()),
            'tipos': df.dtypes.astype(str).to_dict(),
            'nulos_percentual': (100 * df.isna().mean()).to_dict(),
            'duplicatas_exatas': int(df.duplicated().sum()),
            'estatisticas': df.describe(include='number').to_dict() if len(df.select_dtypes('number').columns) else {},
            'categorias': {c: {'nunique': int(categories[c].nunique()),
                               'primeiros_50': categories[c].dropna().unique()[:50].tolist()}
                           for c in categories},
            'periodo_segundos_deslocados': {
                c: [df[c].min(), df[c].max()] for c in ['start_time', 'end_time'] if c in df}}


def ler_selecao(data_dir: Path, tabela: str, cfg: dict, report: dict,
                filtro: Any = None) -> pd.DataFrame:
    """Conta todas as linhas; mantém somente as necessárias para a amostra."""
    pieces, total, empty = [], 0, None
    for chunk in iter_csv(data_dir, tabela, cfg):
        total += len(chunk)
        empty = chunk.iloc[:0]
        for col in chunk.select_dtypes(include=['string']):
            chunk[col] = chunk[col].str.strip().replace('', pd.NA)
        selected = chunk if filtro is None else chunk.loc[filtro(chunk)]
        if len(selected):
            pieces.append(selected.copy())
    if empty is None:
        raise ValueError(f'Arquivo sem registros: {tabela}.')
    df = pd.concat(pieces, ignore_index=True) if pieces else empty
    report[tabela] = {'linhas_totais_lidas': total, 'selecao': perfil(df)}
    LOG.info('%s: %s linhas brutas, %s selecionadas', tabela, total, len(df))
    return df


def dimensao_unica(df: pd.DataFrame, keys: list[str], name: str, report: dict) -> pd.DataFrame:
    """Remove só duplicatas exatas; conflitos de chave exigem decisão explícita."""
    clean = df.drop_duplicates()
    report[name] = {'duplicatas_exatas_removidas': len(df) - len(clean),
                    'chaves_nulas_quarentena': int(clean[keys].isna().any(axis=1).sum())}
    clean = clean.dropna(subset=keys)
    if clean.duplicated(keys).any():
        examples = clean.loc[clean.duplicated(keys, keep=False), keys].head().to_dict('records')
        raise ValueError(f'Cardinalidade ambígua em {name}: {examples}')
    return clean


def juntar(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], name: str,
           report: dict) -> pd.DataFrame:
    """Left join many_to_one auditado, sem multiplicar observações."""
    right = dimensao_unica(right, keys, name, report)
    result = left.merge(right, on=keys, how='left', validate='many_to_one', indicator=True,
                        suffixes=('', f'_{name}'))
    report[name].update({'chaves': keys, 'tipo': 'left', 'cardinalidade': 'many_to_one',
                         'sem_correspondencia': int(result['_merge'].eq('left_only').sum())})
    if len(result) != len(left):
        raise AssertionError('Join multiplicou linhas.')
    return result.drop(columns='_merge')


def resolver_ambiguidades(df: pd.DataFrame, keys: list[str], name: str,
                         report: dict, quarantine: Path) -> pd.DataFrame:
    """Preserva chaves ambíguas, anula seus atributos e exporta evidências brutas."""
    clean = df.drop_duplicates().copy()
    conflict = clean.duplicated(keys, keep=False)
    report[name] = {'duplicatas_exatas_removidas': int(len(df)-len(clean)),
                    'linhas_conflitantes': int(conflict.sum())}
    if not conflict.any():
        clean[f'{name}_ambigua'] = False
        return clean
    quarantine.mkdir(exist_ok=True, parents=True)
    clean.loc[conflict].to_parquet(quarantine / f'{name}.parquet', index=False)
    ambiguous = clean.loc[conflict].drop_duplicates(keys).copy()
    for col in ambiguous.columns.difference(keys):
        ambiguous[col] = np.nan if pd.api.types.is_numeric_dtype(ambiguous[col]) else pd.NA
    ambiguous[f'{name}_ambigua'] = True
    good = clean.loc[~conflict].copy()
    good[f'{name}_ambigua'] = False
    report[name]['identidades_ambiguas'] = len(ambiguous)
    report[name]['quarentena'] = str(quarantine / f'{name}.parquet')
    return pd.concat([good, ambiguous], ignore_index=True)


def carregar_integrar(data_dir: Path, cfg: dict, report: dict, cache: Path) -> pd.DataFrame:
    """Amostra jobs e integra sensores, instâncias, pedidos, tags e máquinas."""
    report['fontes'] = {}
    sources = report['fontes']
    cache.mkdir(exist_ok=True, parents=True)
    # Cache inclui configuração e identidade dos arquivos de origem/schema.
    files = list(data_dir.glob('pai_*')) + list((SCHEMA / 'data').glob('*.header'))
    signature = hashlib.sha256(json.dumps({
        'versao_cache': 3, 'fracao': cfg['fracao_amostra'], 'seed': cfg['seed'],
        'fontes': [(str(f), f.stat().st_size, f.stat().st_mtime_ns) for f in sorted(files)]
    }, sort_keys=True).encode()).hexdigest()[:20]
    cp, rp = cache / f'{signature}.parquet', cache / f'{signature}.json'
    if cp.exists() and rp.exists():
        report.update({k: v for k, v in json.loads(rp.read_text()).items()
                       if k not in {'configuracao', 'schema_fonte', 'aviso'}})
        LOG.info('Usando integração em cache: %s', cp)
        return pd.read_parquet(cp)
    def read(tabela: str, filtro: Any = None) -> pd.DataFrame:
        """Persiste seleções concluídas para retomar após falha de qualidade."""
        path = cache / f'{signature}_{tabela}.parquet'
        meta = path.with_suffix('.json')
        if path.exists() and meta.exists():
            sources[tabela] = json.loads(meta.read_text())
            LOG.info('Seleção em cache: %s', tabela)
            return pd.read_parquet(path)
        result = ler_selecao(data_dir, tabela, cfg, sources, filtro)
        result.to_parquet(path, index=False)
        salvar_json(meta, sources[tabela])
        return result

    sensors = read('pai_sensor_table',
                          lambda x: amostrar_jobs(x['inst_id'], cfg))
    instances = read('pai_instance_table',
                            lambda x: amostrar_jobs(x['inst_id'], cfg))
    if instances.empty:
        raise ValueError('Amostra de instâncias vazia; aumente fracao_amostra.')
    jobs = set(instances['inst_id'].dropna())
    names = set(instances['job_name'].dropna())
    tasks = read('pai_task_table',
                        lambda x: x['job_name'].isin(names))
    tags = read('pai_group_tag_table',
                       lambda x: x['inst_id'].isin(jobs))
    specs = read('pai_machine_spec')
    # Inspeção das outras tabelas, sem integrar médias de máquina redundantes.
    job_table = read('pai_job_table',
                            lambda x: x['inst_id'].isin(jobs))
    workers = set(instances['worker_name'].dropna())
    machine_metrics = read('pai_machine_metric',
                                  lambda x: x['worker_name'].isin(workers))
    del job_table, machine_metrics
    report['joins'] = {}
    audit = report['joins']
    report['ambiguidades'] = {}
    report['linhas_antes_integracao'] = len(instances)
    quarantine = cache / f'quarentena_{signature}'
    # Identidades incompletas não podem ser integradas; guardar os originais
    # antes de resolver duplicatas para não perder tentativas sem máquina.
    null_instances = instances.loc[instances[IDENTIDADE].isna().any(axis=1)]
    if len(null_instances):
        quarantine.mkdir(exist_ok=True, parents=True)
        null_instances.to_parquet(quarantine / 'instancias_chaves_nulas.parquet', index=False)
    instances = resolver_ambiguidades(instances, IDENTIDADE, 'instancias',
        report['ambiguidades'], quarantine)
    if len(null_instances):
        report['ambiguidades']['instancias']['quarentena_chaves_nulas'] = str(
            quarantine / 'instancias_chaves_nulas.parquet')
        report['ambiguidades']['instancias']['linhas_originais_chaves_nulas'] = len(null_instances)
    tasks = resolver_ambiguidades(tasks[['job_name', 'task_name', 'plan_cpu', 'plan_gpu', 'plan_mem', 'gpu_type']],
        ['job_name', 'task_name'], 'tarefas', report['ambiguidades'], quarantine)
    report['sensores_duplicatas_exatas_removidas'] = int(sensors.duplicated().sum())
    sensors = sensors.drop_duplicates()
    # Dados são por worker: CPU/memória nunca se somam entre GPUs.
    report['sensores_chaves_nulas_quarentena'] = int(sensors[IDENTIDADE].isna().any(axis=1).sum())
    sensors = sensors.dropna(subset=IDENTIDADE)
    # Uma linha de sensor por worker no trace; várias linhas por dispositivo são
    # aceitas apenas quando cada gpu_name é distinto e não há inconsistência.
    if sensors.duplicated(IDENTIDADE + ['gpu_name']).any():
        raise ValueError('Sensor repetido com valores conflitantes para worker/dispositivo.')
    grouped = sensors.groupby(IDENTIDADE, dropna=False, sort=False)
    sensor = grouped.agg(cpu_usage=('cpu_usage', 'mean'), avg_mem=('avg_mem', 'mean'),
                         gpu_wrk_util=('gpu_wrk_util', lambda s: s.sum(min_count=1)),
                         dispositivos_observados=('gpu_name', 'nunique')).reset_index()
    # A instância é a âncora: também preserva instâncias sem sensores no job amostrado.
    base = dimensao_unica(instances, IDENTIDADE, 'instancias', audit)
    base = juntar(base, sensor, IDENTIDADE, 'sensores', audit)
    base = juntar(base, tasks,
                  ['job_name', 'task_name'], 'tarefas', audit)
    base = juntar(base, specs, ['machine'], 'maquinas', audit)
    base = juntar(base, tags[['inst_id', 'workload']], ['inst_id'], 'tags', audit)
    report['linhas_depois_integracao'] = len(base)
    report['colunas_agregadas_ou_nao_integradas_sensor'] = [
        c for c in sources['pai_sensor_table']['selecao']['tipos'] if c not in base]
    base.to_parquet(cp, index=False)
    salvar_json(rp, {k: v for k, v in report.items()
                     if k not in {'configuracao', 'schema_fonte', 'aviso'}})
    return base


def razao(num: pd.Series, den: pd.Series) -> pd.Series:
    """Divisão definida apenas para denominador estritamente positivo."""
    return num / den.where(den.gt(0))


def atributos(df: pd.DataFrame, cfg: dict, report: dict) -> pd.DataFrame:
    """Padroniza unidades e gera proxies sem imputar medições para o alvo."""
    df = df.copy()
    for col in ['status', 'task_name', 'gpu_type', 'workload']:
        df[col] = df[col].astype('string').str.strip().str.lower().replace('', pd.NA)
    df['inicio'] = pd.to_datetime(df['start_time'].where(df['start_time'].gt(0)), unit='s', utc=True).dt.tz_convert('Asia/Shanghai')
    df['fim'] = pd.to_datetime(df['end_time'].where(df['end_time'].gt(0)), unit='s', utc=True).dt.tz_convert('Asia/Shanghai')
    df['duracao_job'] = (df['fim'] - df['inicio']).dt.total_seconds()
    df['intervalo_invalido'] = df['inicio'].isna() | df['fim'].isna() | df['duracao_job'].lt(0)
    # Não inferir servidor inativo a partir de uma instância sem intervalo.
    df['servidor_ativo'] = pd.Series(pd.NA, index=df.index, dtype='boolean')
    df.loc[~df['intervalo_invalido'] & df['machine'].notna(), 'servidor_ativo'] = True
    df['janela_inicio'] = df['inicio'].dt.floor(cfg['janela'])
    df['hora_do_dia'] = df['inicio'].dt.hour
    df['dia_da_semana'] = df['inicio'].dt.day_of_week
    df['fim_de_semana'] = df['dia_da_semana'].ge(5).where(df['inicio'].notna()).astype('Float64')
    df['turno'] = pd.cut(df['hora_do_dia'], [-1, 5, 11, 17, 23], labels=['madrugada', 'manha', 'tarde', 'noite']).astype('string')
    df['indicador_pico'] = df['hora_do_dia'].between(*cfg['pico_horas'], inclusive='left').where(df['inicio'].notna()).astype('Float64')
    for raw, dest, divisor in [('plan_gpu', 'recursos_gpu_alocados', 100),
                                ('gpu_wrk_util', 'recursos_gpu_utilizados', 100),
                                ('plan_cpu', 'recursos_cpu_alocados', 100),
                                ('cpu_usage', 'recursos_cpu_utilizados', 100),
                                ('plan_mem', 'memoria_alocada', 1), ('avg_mem', 'memoria_utilizada', 1)]:
        df[dest] = df[raw] / divisor
        df[f'{dest}_invalido'] = df[dest].lt(0) | ~np.isfinite(df[dest].fillna(0))
    pairs = [('gpu', 'recursos_gpu_utilizados', 'recursos_gpu_alocados'),
             ('cpu', 'recursos_cpu_utilizados', 'recursos_cpu_alocados'),
             ('memoria', 'memoria_utilizada', 'memoria_alocada')]
    invalid = pd.Series(False, index=df.index)
    for resource, used, allocated in pairs:
        df[f'utilizacao_{resource}'] = 100 * razao(df[used], df[allocated])
        df[f'utilizacao_{resource}_fora_intervalo'] = ~df[f'utilizacao_{resource}'].between(0, 100) & df[f'utilizacao_{resource}'].notna()
        valid_used = df[used].mask(df[f'{used}_invalido'])
        valid_alloc = df[allocated].mask(df[f'{allocated}_invalido'])
        # Sobreuso acima do pedido: preserva valor bruto, subutilização limitada a 0.
        rate = razao(valid_used, valid_alloc)
        df[f'taxa_utilizacao_{resource}'] = rate
        df[f'taxa_subutilizacao_{resource}'] = (1 - rate).clip(0, 1)
        df[f'desperdicio_{resource}'] = df[allocated] - df[used]
        invalid |= df[f'{used}_invalido'] | df[f'{allocated}_invalido']
    df['eficiencia_alocacao_gpu'] = df['taxa_utilizacao_gpu']
    df['participacao_cpu_capacidade_maquina'] = 100 * razao(df['recursos_cpu_utilizados'], df['cap_cpu'])
    df['participacao_memoria_capacidade_maquina'] = 100 * razao(df['memoria_utilizada'], df['cap_mem'])
    df['participacao_gpu_capacidade_maquina'] = 100 * razao(df['recursos_gpu_utilizados'], df['cap_gpu'])
    df['gpu_ociosa'] = df['utilizacao_gpu'].lt(cfg['gpu_ociosa_limiar']).where(df['utilizacao_gpu'].notna()).astype('boolean')
    df['gpu_alocada_ociosa'] = (df['recursos_gpu_alocados'].gt(0) & df['gpu_ociosa']).astype('boolean')
    df['utilizacao_gpu_baixa'] = df['gpu_ociosa']
    df['cpu_subutilizada'] = df['utilizacao_cpu'].lt(cfg['cpu_baixa_limiar']).where(df['utilizacao_cpu'].notna()).astype('boolean')
    df['recurso_solicitado_muito_acima_do_utilizado'] = df['eficiencia_alocacao_gpu'].lt(1 / cfg['razao_solicitado_utilizado']).where(df['eficiencia_alocacao_gpu'].notna()).astype('boolean')
    df['falha_instancia'] = df['status'].eq('failed').where(df['status'].notna()).astype('boolean')
    sub = [f'taxa_subutilizacao_{x}' for x in ['gpu', 'cpu', 'memoria']]
    df['intensidade_desperdicio'] = df[sub].mean(axis=1, skipna=False)
    weights = cfg['pesos']
    score = sum(df[f'taxa_subutilizacao_{k}'] * v for k, v in weights.items()) / sum(weights.values())
    df['score_desperdicio'] = score.mask(invalid | df['intervalo_invalido'])
    df['classe_risco_desperdicio'] = pd.cut(df['score_desperdicio'],
        [-np.inf, *cfg['limiares_classes'], np.inf], labels=CLASSES, right=False).astype('string')
    df['motivo_sem_rotulo'] = pd.Series(pd.NA, index=df.index, dtype='string')
    reasons = np.select([
        invalid.to_numpy(), df['intervalo_invalido'].to_numpy(),
        df['recursos_gpu_alocados'].fillna(0).le(0).to_numpy(),
        df[['recursos_gpu_utilizados','recursos_cpu_utilizados','memoria_utilizada']].isna().any(axis=1).to_numpy(),
    ], ['valor_negativo_ou_nao_finito', 'intervalo_ambiguo_ou_invalido',
        'pedido_gpu_ausente_ou_nao_positivo', 'medicao_ausente'], default='pedido_cpu_ou_memoria_ausente_ou_nao_positivo')
    missing_label = df['classe_risco_desperdicio'].isna()
    df.loc[missing_label, 'motivo_sem_rotulo'] = reasons[missing_label]
    report['atributos'] = {'unidade': 'instancia_job_vida_observada',
        'periodo_ficticio': [str(df['inicio'].min()), str(df['fim'].max())],
        'nulos_percentual': (df.isna().mean() * 100).to_dict(),
        'estatisticas_proxies': df[['utilizacao_gpu', 'utilizacao_cpu', 'utilizacao_memoria',
            'duracao_job', 'recursos_gpu_alocados', 'recursos_cpu_alocados',
            'memoria_alocada', 'score_desperdicio']].describe().to_dict(),
        'flags_inconsistencia': {c: int(df[c].sum()) for c in df if c.endswith(('_invalido', '_fora_intervalo'))},
        'sem_rotulo': int(df['classe_risco_desperdicio'].isna().sum()),
        'motivos_sem_rotulo': df['motivo_sem_rotulo'].value_counts().to_dict(),
        'distribuicao_classes': df['classe_risco_desperdicio'].value_counts().to_dict(),
        'colunas_criadas': [c for c in df if c not in set(IDENTIDADE + list(NUMERICAS))]}
    return df.replace([np.inf, -np.inf], np.nan)


def componentes_servidores(df: pd.DataFrame) -> pd.Series:
    """Componentes job-máquina evitam que um job multi-servidor atravesse splits."""
    parent: dict[str, str] = {}
    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for job, machine in df[['inst_id', 'machine']].drop_duplicates().itertuples(index=False, name=None):
        a, b = find('j:' + str(job)), find('m:' + str(machine))
        parent[a] = b
    return df['machine'].map(lambda m: find('m:' + str(m)))


def estratificar_grupos(df: pd.DataFrame, group: pd.Series, fraction: float,
                        cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Escolhe fold com proporção e distribuição mais próximas, mantendo grupos."""
    n = min(max(2, round(1 / fraction)), int(group.nunique()))
    if n < 2:
        raise ValueError('Menos de dois grupos independentes para divisão.')
    cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=cfg['seed'])
    y = df['classe_risco_desperdicio']
    base_dist = y.value_counts(normalize=True).reindex(CLASSES, fill_value=0)
    candidates = list(cv.split(df, y, groups=group))
    def cost(pair: tuple) -> float:
        test = pair[1]
        dist = y.iloc[test].value_counts(normalize=True).reindex(CLASSES, fill_value=0)
        return abs(len(test) / len(df) - fraction) + (dist - base_dist).abs().mean()
    return min(candidates, key=cost)


def dividir(df: pd.DataFrame, cfg: dict, report: dict) -> pd.DataFrame:
    """Divide antes de ajustar qualquer transformação estatística."""
    df = df.copy()
    df['conjunto'] = 'sem_rotulo'
    eligible = df.loc[df['classe_risco_desperdicio'].notna()].sort_values('inicio')
    if len(eligible) < 30:
        raise ValueError('Poucos registros rotulados; aumente a amostra.')
    a, b, c = cfg['proporcoes']
    purged = 0
    if cfg['split'] == 'temporal':
        times = eligible['inicio'].drop_duplicates().sort_values()
        if len(times) < 3:
            raise ValueError('São necessários três timestamps distintos.')
        t1 = times.iloc[min(len(times)-2, max(1, int(len(times) * a)))]
        t2 = times.iloc[min(len(times)-1, max(2, int(len(times) * (a+b))))]
        train = eligible[(eligible['inicio'] < t1) & (eligible['fim'] < t1)]
        val = eligible[(eligible['inicio'] >= t1) & (eligible['inicio'] < t2) & (eligible['fim'] < t2)]
        test = eligible[eligible['inicio'] >= t2]
        # Todas as linhas de um job que toca o conjunto seguinte são purgadas.
        later_jobs = set(eligible.loc[eligible['inicio'].ge(t1), 'inst_id'])
        train = train[~train['inst_id'].isin(later_jobs)]
        test_jobs = set(test['inst_id'])
        val = val[~val['inst_id'].isin(test_jobs)]
        pieces = [train.index, val.index, test.index]
        purged = len(eligible) - sum(map(len, pieces))
        df.loc[eligible.index, 'conjunto'] = 'purgado'
        report['fronteiras_temporais'] = [str(t1), str(t2)]
    else:
        group = eligible['inst_id'] if cfg['split'] == 'estratificado' else componentes_servidores(eligible)
        first, test_idx = estratificar_grupos(eligible, group, c, cfg)
        development = eligible.iloc[first]
        train_idx, val_idx = estratificar_grupos(development, group.iloc[first], b / (a+b), cfg)
        pieces = [development.iloc[train_idx].index, development.iloc[val_idx].index,
                  eligible.iloc[test_idx].index]
    for name, index in zip(['treino', 'validacao', 'teste'], pieces):
        if not len(index):
            raise ValueError(f'Split {name} vazio. Aumente a amostra ou use outro split.')
        df.loc[index, 'conjunto'] = name
    jobsets = [set(df.loc[df['conjunto'].eq(s), 'inst_id']) for s in ['treino','validacao','teste']]
    if any(jobsets[i] & jobsets[j] for i in range(3) for j in range(i+1, 3)):
        raise AssertionError('Vazamento: job em mais de um conjunto.')
    if cfg['split'] == 'servidores':
        sets = [set(df.loc[df['conjunto'].eq(s), 'machine']) for s in ['treino','validacao','teste']]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i+1, 3)):
            raise AssertionError('Vazamento de servidor.')
    report['divisao'] = {'metodo': cfg['split'], 'purgados': purged,
        'contagens': df['conjunto'].value_counts().to_dict(),
        'classes_por_conjunto': pd.crosstab(df['conjunto'], df['classe_risco_desperdicio']).to_dict(),
        'observacao': 'Proporções aproximadas: grupos e fronteiras temporais têm precedência.'}
    return df


def sinalizar_outliers(df: pd.DataFrame, cfg: dict, report: dict) -> pd.DataFrame:
    """Aprende IQR, Z-score e percentis só no treino e preserva observações."""
    df = df.copy()
    train = df.loc[df['conjunto'].eq('treino')]
    thresholds, counts = {}, {}
    for col in ['utilizacao_gpu', 'utilizacao_cpu', 'utilizacao_memoria', 'duracao_job']:
        values = train[col].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([.25, .75])
        lo, hi = q1 - cfg['iqr_fator'] * (q3-q1), q3 + cfg['iqr_fator'] * (q3-q1)
        mean, std = values.mean(), values.std(ddof=0)
        p1, p99 = values.quantile(cfg['percentis_outliers'])
        flags = {'iqr': df[col].lt(lo) | df[col].gt(hi),
                 'zscore': ((df[col] - mean).abs() / std).gt(cfg['z_limiar']) if std > 0 else pd.Series(False, index=df.index),
                 'percentil': df[col].lt(p1) | df[col].gt(p99)}
        for method, mask in flags.items():
            name = f'{col}_outlier_{method}'
            df[name] = mask.where(df[col].notna()).astype('boolean')
            counts[name] = int(mask.sum())
        thresholds[col] = {'iqr': [lo, hi], 'media': mean, 'desvio': std, 'percentis': [p1, p99]}
    df['utilizacao_gpu_extrema'] = df['utilizacao_gpu_outlier_iqr']
    p1, p99 = thresholds['duracao_job']['percentis']
    df['job_longa_duracao'] = df['duracao_job'].gt(p99).where(df['duracao_job'].notna()).astype('boolean')
    df['job_curta_duracao'] = df['duracao_job'].lt(p1).where(df['duracao_job'].notna()).astype('boolean')
    report['outliers'] = {'limiares_treino': thresholds, 'contagens': counts,
                          'removidos': 0, 'aviso': 'Picos podem ser normais em treinamento de IA.'}
    return df


def montar_preprocessador(train: pd.DataFrame, cfg: dict, report: dict) -> tuple[Pipeline, list[str]]:
    """Escolhe colunas no treino e retorna ColumnTransformer/Pipeline sklearn."""
    cols = PREDITORES_NUM + PREDITORES_CAT
    constants = [x for x in cols if train[x].nunique(dropna=False) <= 1]
    num = [x for x in PREDITORES_NUM if x not in constants]
    cat = [x for x in PREDITORES_CAT if x not in constants]
    corr = train[num].corr()
    redundant = [c for i, c in enumerate(num) if any(abs(corr.loc[c, p]) >= cfg['correlacao_limiar'] for p in num[:i])]
    num = [x for x in num if x not in redundant]
    if not num and not cat:
        raise ValueError('Nenhum preditor variável no treino.')
    columns = ColumnTransformer([
        ('numericas', Pipeline([('imputar', SimpleImputer(strategy='median', add_indicator=True)),
                                ('escalonar', StandardScaler())]), num),
        ('categoricas', Pipeline([('imputar', SimpleImputer(strategy='constant', fill_value='Desconhecido')),
                                  ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat)
    ], remainder='drop', verbose_feature_names_out=True)
    steps: list[tuple[str, Any]] = [('colunas', columns), ('remover_constantes', VarianceThreshold())]
    if cfg['selecao'] == 'kbest':
        # k é limitado depois que o encoder for ajustado, na função treinar.
        steps.append(('selecao', SelectKBest(f_classif, k='all')))
    elif cfg['selecao'] == 'forest':
        steps.append(('selecao', SelectFromModel(RandomForestClassifier(n_estimators=100,
                      class_weight='balanced', random_state=cfg['seed'], n_jobs=1))))
    elif cfg['selecao'] == 'pca':
        steps.append(('selecao', PCA(n_components=cfg['pca_variancia'], svd_solver='full')))
    report['selecao'] = {'constantes_removidas': constants, 'redundantes_removidas': redundant,
        'preditores_mantidos': num + cat, 'metodo': cfg['selecao'],
        'justificativa': 'Pedidos, papel da task, GPU e calendário disponíveis ao iniciar; sem componentes do alvo.',
        'excluidos_por_vazamento': ['score_desperdicio', 'utilizacao_*', 'desperdicio_*', 'duracao_job', 'status', 'falha_instancia'],
        'correlacoes_preditores_treino': corr.to_dict()}
    return Pipeline(steps), num + cat


def treinar(df: pd.DataFrame, cfg: dict, out: Path, report: dict) -> None:
    """Ajusta transformações e baseline só no treino; salva artefatos reutilizáveis."""
    train = df[df['conjunto'].eq('treino')]
    preprocess, predictors = montar_preprocessador(train, cfg, report)
    # sklearn espera np.nan para ausências, não pd.NA das strings pandas.
    x = df[predictors].astype(object).where(df[predictors].notna(), np.nan)
    y_train = train['classe_risco_desperdicio'].astype(str)
    if cfg['selecao'] == 'kbest':
        temp = Pipeline(preprocess.steps[:-1])
        encoded = temp.fit_transform(x.loc[train.index], y_train)
        preprocess.set_params(selecao__k=min(cfg['k_best'], encoded.shape[1]))
    preprocess.fit(x.loc[train.index], y_train)
    report['selecao']['atributos_transformados'] = preprocess.get_feature_names_out().tolist()
    joblib.dump(preprocess, out / 'preprocessador.joblib')
    for name in ['treino', 'validacao', 'teste']:
        part = df[df['conjunto'].eq(name)]
        transformed = preprocess.transform(x.loc[part.index])
        np.savez_compressed(out / f'{name}_transformado.npz', X=transformed,
                           y=part['classe_risco_desperdicio'].astype(str).to_numpy(dtype=str),
                           indice=part.index.to_numpy())
    if cfg['treinar_baseline']:
        model = Pipeline([('preprocessamento', preprocess), ('classificador', RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=3,
            class_weight='balanced', random_state=cfg['seed'], n_jobs=1))])
        model.fit(x.loc[train.index], y_train)
        report['baseline'] = {}
        for name in ['validacao', 'teste']:
            part = df[df['conjunto'].eq(name)]
            pred = model.predict(x.loc[part.index])
            report['baseline'][name] = classification_report(part['classe_risco_desperdicio'], pred,
                labels=CLASSES, output_dict=True, zero_division=0)
            report['baseline'][name]['balanced_accuracy'] = float(
                balanced_accuracy_score(part['classe_risco_desperdicio'], pred))
        report['baseline']['aviso'] = 'Avalia rótulo heurístico retrospectivo, não risco ambiental real.'
        joblib.dump(model, out / 'modelo_baseline.joblib')
        if cfg['selecao'] == 'nenhuma':
            names = preprocess.get_feature_names_out()
            pd.DataFrame({'atributo': names, 'importancia': model[-1].feature_importances_}).sort_values(
                'importancia', ascending=False).to_csv(out / 'importancia_variaveis.csv', index=False)
    report['nulos_tratados_modelo'] = {
        'treino': train[predictors].isna().sum().to_dict(),
        'estrategia': 'Mediana numérica e categoria Desconhecido; ajuste somente no treino. Base auditável preserva nulos.'}
    report['desbalanceamento'] = {
        'classes_ausentes_treino': sorted(set(CLASSES) - set(y_train)),
        'sugestao': 'class_weight=balanced no baseline; reamostragem somente dentro de folds de treino. Usar StratifiedGroupKFold por job.'}


def graficos_relatorio(df: pd.DataFrame, cfg: dict, out: Path, report: dict) -> None:
    """Gera inspeção retrospectiva e relatório sem usar teste para ajustar dados."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    figures = out / 'graficos'
    figures.mkdir(exist_ok=True)
    sns.set_theme(style='whitegrid')
    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(figures / f'{name}.png', dpi=140)
        plt.close()
    for col in ['utilizacao_gpu', 'utilizacao_cpu', 'utilizacao_memoria', 'duracao_job', 'recursos_gpu_alocados']:
        plt.figure(figsize=(9, 4))
        sns.histplot(df[col].dropna(), bins=45)
        plt.title(f'Distribuição: {col} (médias de vida)')
        if col.startswith('utilizacao_'):
            plt.xlabel('Uso médio / recurso solicitado (%)')
        save(f'histograma_{col}')
    plt.figure(figsize=(9, 4))
    plt.boxplot([df.loc[df['classe_risco_desperdicio'].eq(c), 'utilizacao_gpu'].dropna().to_numpy()
                 for c in CLASSES], tick_labels=CLASSES)
    plt.xlabel('Classe heurística de risco'); plt.ylabel('Uso médio / GPU solicitada (%)')
    plt.title('Uso relativo ao pedido — valores acima de 100% indicam sobreuso')
    save('boxplot_gpu_por_classe')
    plt.figure(figsize=(8, 4))
    df['classe_risco_desperdicio'].value_counts().reindex(CLASSES, fill_value=0).plot.bar()
    plt.ylabel('Instâncias'); plt.title('Distribuição dos rótulos heurísticos'); save('distribuicao_classes')
    numeric = ['utilizacao_gpu','utilizacao_cpu','utilizacao_memoria','duracao_job',
               'recursos_gpu_alocados','recursos_cpu_alocados','memoria_alocada','score_desperdicio']
    corr = df.loc[df['conjunto'].eq('treino'), numeric].corr()
    corr.to_csv(out / 'correlacao_treino.csv')
    report['correlacoes_relevantes_treino'] = [
        {'a': a, 'b': b, 'correlacao': float(corr.loc[a,b])}
        for i, a in enumerate(numeric) for b in numeric[i+1:]
        if pd.notna(corr.loc[a,b]) and abs(corr.loc[a,b]) >= cfg['correlacao_limiar']]
    plt.figure(figsize=(10, 7)); sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, annot=True, fmt='.2f')
    plt.title('Correlação no conjunto de treino')
    save('matriz_correlacao_treino')
    machines = df.loc[df['utilizacao_gpu'].notna() & df['cap_gpu'].gt(0), 'machine'].value_counts().head(3).index
    cohort = df[df['machine'].isin(machines)].groupby(['machine','janela_inicio'])['utilizacao_gpu'].mean().reset_index()
    plt.figure(figsize=(12, 4))
    for machine in machines:
        subset = cohort[cohort['machine'].eq(machine)]
        plt.plot(subset['janela_inicio'], subset['utilizacao_gpu'], label=str(machine)[:10], marker='.', linewidth=.6)
    plt.legend(); plt.title('Coortes por início — médias de vida, sem resolução intrajob')
    plt.xlabel('Datas deslocadas (Asia/Shanghai)'); plt.ylabel('Uso médio / GPU solicitada (%)')
    plt.xticks(rotation=25); save('serie_coortes_servidor')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (used, allocated, unit) in zip(axes, [
        ('recursos_gpu_utilizados','recursos_gpu_alocados','GPUs equivalentes'),
        ('recursos_cpu_utilizados','recursos_cpu_alocados','núcleos'),
        ('memoria_utilizada','memoria_alocada','GB')]):
        df[[allocated, used]].mean().plot.bar(ax=ax)
        ax.set_title(unit); ax.tick_params(axis='x', rotation=25)
    save('alocados_vs_utilizados')
    # Agregado de coorte é descritivo; NÃO jobs ativos ou utilização temporal real.
    cohort_all = df.groupby(['machine','janela_inicio']).agg(
        instancias_iniciadas=('worker_name','size'), jobs_iniciados=('inst_id','nunique'),
        gpu_media_vida_media=('utilizacao_gpu','mean'), gpu_media_vida_max=('utilizacao_gpu','max'),
        gpu_media_vida_min=('utilizacao_gpu','min'), cpu_media_vida_media=('utilizacao_cpu','mean'),
        memoria_media_vida_media=('utilizacao_memoria','mean'),
        soma_gpu_solicitada_instancias=('recursos_gpu_alocados',lambda s: s.sum(min_count=1)),
        soma_duracao_instancias_s=('duracao_job',lambda s: s.sum(min_count=1))).reset_index()
    cohort_all.to_parquet(out / 'coortes_inicio.parquet', index=False)
    report['final'] = {'linhas': len(df), 'colunas': len(df.columns),
                        'memoria_bytes': int(df.memory_usage(deep=True).sum()),
                        'duplicatas_finais_identidade': int(df.duplicated(IDENTIDADE).sum())}
    salvar_json(out / 'relatorio_qualidade.json', report)
    counts = df['classe_risco_desperdicio'].value_counts().reindex(CLASSES, fill_value=0)
    lines = [
        '# Relatório de qualidade — Alibaba GPU v2020', '',
        '**Rótulo heurístico de recursos; não é medição de energia ou risco ambiental validado.**', '',
        f"Fração de jobs selecionados por hash: {cfg['fracao_amostra']:.2%}; seed={cfg['seed']}.",
        f"Base final: {len(df):,} linhas, {len(df.columns)} colunas. Unidade: instância durante sua vida observada.",
        f"Instâncias antes da integração: {report['linhas_antes_integracao']:,}; depois: {report['linhas_depois_integracao']:,}.",
        f"Período fictício: {report['atributos']['periodo_ficticio']} (Asia/Shanghai).",
        f"Sem rótulo: {report['atributos']['sem_rotulo']}; mantidos para auditoria.", '',
        f"Motivos: {report['atributos']['motivos_sem_rotulo']}",
        f"Ambiguidades e quarentena: {report['ambiguidades']}", '',
        '## Classes', '', *[f'- {c}: {int(counts[c]):,}' for c in CLASSES], '',
        '## Fontes e integrações', '',
        *[f"- {t}: {v['linhas_totais_lidas']:,} linhas lidas; {v['selecao']['linhas']:,} na seleção." for t,v in report['fontes'].items()], '',
        f"Joins auditados: `{json.dumps(report['joins'], ensure_ascii=False)}`", '',
        '## Treino e validação', '',
        f"Divisão: {report['divisao']}",
        f"Remoção de constantes: {report['selecao']['constantes_removidas']}",
        f"Remoção de redundantes: {report['selecao']['redundantes_removidas']}",
        f"Preditores: {report['selecao']['preditores_mantidos']}",
        'Imputação, escala, seleção, outliers e correlações ajustados só no treino.',
        'class_weight=balanced no baseline. Classes heurísticas não constituem validação ambiental.', '',
        '## Limitações', '',
        'Médias de vida não permitem reconstruir janelas de atividade, médias móveis de 15min, variação de 1h ou simultaneidade.',
        'Percentuais são relativos ao pedido. Valores acima de 100% são preservados e sinalizados.',
        'Não há identificação de rack, potência, kWh, temperatura ou refrigeração. Ausência não equivale a ociosidade.',
        'A amostra por jobs preserva observações relacionadas, mas não garante representatividade por máquina/tipo raro.',
        'Outliers são sinalizados, não removidos. Picos podem ser normais em IA.',
        'Estatísticas descritivas das fontes são da seleção; linhas totais resultam da leitura completa.',
        'Consulte relatorio_qualidade.json para nulos, tipos, memória, categorias, duplicatas, outliers e limiares.',
        'Consulte DECISOES.md para schema, unidades e justificativas.', '',
        '## Algoritmos sugeridos', '',
        'Random Forest é o baseline incluído. Compare XGBoost/LightGBM para interações tabulares; SVM para amostras menores escalonadas.',
        'Valide limiares com especialistas e dados de potência antes de interpretar previsões como impacto ambiental.', '',
        '## Gráficos', '', *[f'![{p.stem}](graficos/{p.name})' for p in sorted(figures.glob('*.png'))]]
    (out / 'relatorio_qualidade.md').write_text('\n'.join(lines), encoding='utf-8')


def executar(data_dir: Path, out: Path, cfg: dict) -> dict:
    """Executa integração, features, divisão, ajuste no treino e exportações."""
    validar_config(cfg)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {'configuracao': cfg, 'schema_fonte': str(SCHEMA), 'aviso': 'Rótulos heurísticos retrospectivos.'}
    base = carregar_integrar(data_dir, cfg, report, data_dir / 'cache')
    for details in report.get('ambiguidades', {}).values():
        for key in [k for k in details if k.startswith('quarentena')]:
            dest = out / 'quarentena'
            dest.mkdir(exist_ok=True)
            source = Path(details[key])
            shutil.copy2(source, dest / source.name)
            details[key] = str(dest / source.name)
    report['linhas_antes_features'] = len(base)
    final = atributos(base, cfg, report)
    final = dividir(final, cfg, report)
    final = sinalizar_outliers(final, cfg, report)
    treinar(final, cfg, out, report)
    final.to_csv(out / 'base_processada.csv', index=False)
    final.to_parquet(out / 'base_processada.parquet', index=False)
    # Treino/validação/teste tabulares mantêm identificadores só para auditoria.
    for name in ['treino','validacao','teste']:
        final[final['conjunto'].eq(name)].to_parquet(out / f'{name}.parquet', index=False)
    salvar_json(out / 'configuracao.json', cfg)
    graficos_relatorio(final, cfg, out, report)
    descrever_colunas(final, out, report)
    LOG.info('Concluído: %s linhas; resultados em %s', len(final), out)
    return report


def descrever_colunas(df: pd.DataFrame, out: Path, report: dict) -> None:
    """Exporta tipos, papéis e percentuais ausentes sem transformar identificadores."""
    rows = []
    predictors = report['selecao']['preditores_mantidos']
    for col in df:
        role = ('alvo_heuristico' if col == 'classe_risco_desperdicio' else
                'preditor' if col in predictors else
                'identificador' if col in IDENTIDADE + ['inst_name'] else
                'data_hora_deslocada' if pd.api.types.is_datetime64_any_dtype(df[col]) else
                'flag_auditoria' if pd.api.types.is_bool_dtype(df[col]) else
                'auxiliar_auditoria')
        rows.append({'coluna': col, 'tipo': str(df[col].dtype), 'papel': role,
                     'nulos_percentual': float(100 * df[col].isna().mean()),
                     'valores_unicos': int(df[col].nunique())})
    pd.DataFrame(rows).to_csv(out / 'dicionario_dados.csv', index=False)


def main() -> None:
    """Interface de linha de comando com falhas explícitas e log em português."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--dados', type=Path, default=ROOT / 'dados')
    parser.add_argument('--saida', type=Path, default=ROOT / 'resultados')
    parser.add_argument('--fracao', type=float)
    parser.add_argument('--split', choices=['temporal','estratificado','servidores'])
    args = parser.parse_args()
    cfg = json.loads(json.dumps(CONFIG))
    if args.config:
        extra = json.loads(args.config.read_text(encoding='utf-8'))
        if unknown := set(extra).difference(CONFIG):
            parser.error(f'Chaves de configuração desconhecidas: {sorted(unknown)}')
        cfg.update(extra)
    if args.fracao is not None:
        cfg['fracao_amostra'] = args.fracao
    if args.split:
        cfg['split'] = args.split
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        validar_config(cfg)
        if args.download:
            download_all(args.dados)
        executar(args.dados, args.saida, cfg)
    except (ValueError, OSError, KeyError) as exc:
        LOG.error('Execução interrompida: %s', exc)
        raise

if __name__ == '__main__':
    main()

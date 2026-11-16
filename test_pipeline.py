"""Testes de regras, cardinalidade e isolamento estatístico, com fixtures sintéticas.

Fixtures são apenas testes; nunca são entregues como dados do Alibaba.
"""
from __future__ import annotations
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline import (CONFIG, IDENTIDADE, SCHEMA, atributos, dimensao_unica, dividir,
                      executar, montar_preprocessador, resolver_ambiguidades,
                      salvar_json, sinalizar_outliers, treinar, validar_config)


def fixture(n: int = 240) -> pd.DataFrame:
    """Gera jobs completos com quatro perfis e tempos monotônicos para testes."""
    rows = []
    for i in range(n):
        rate = [.95, .65, .35, .05][(i // 2) % 4]
        rows.append(dict(job_name=f'job{i//2}', task_name='worker', worker_name=f'w{i}',
            inst_id=f'id{i//2}', machine=f'm{i//20}', inst_name=f'i{i}', status='Terminated',
            start_time=86400.0 + (i//2) * 3600, end_time=86400.0 + (i//2) * 3600 + 900,
            plan_gpu=100., plan_cpu=400. + (i % 3) * 100, plan_mem=16.,
            gpu_wrk_util=100 * rate, cpu_usage=(400 + (i%3)*100) * rate,
            avg_mem=16 * rate, cap_cpu=64., cap_mem=256., cap_gpu=8., gpu_type='V100',
            workload='bert', dispositivos_observados=1))
    return pd.DataFrame(rows)


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = copy.deepcopy(CONFIG)
        self.df = atributos(fixture(), self.cfg, {})

    def test_quatro_classes_e_unidades(self) -> None:
        self.assertEqual(set(self.df.classe_risco_desperdicio), {'Baixo','Moderado','Alto','Crítico'})
        self.assertAlmostEqual(self.df.loc[0, 'recursos_cpu_alocados'], 4)
        self.assertAlmostEqual(self.df.loc[0, 'score_desperdicio'], .05)
        self.assertEqual(str(self.df.inicio.dt.tz), 'Asia/Shanghai')

    def test_nulo_nao_vira_ocioso_ou_rotulo(self) -> None:
        raw = fixture()
        raw.loc[0, 'gpu_wrk_util'] = np.nan
        frame = atributos(raw, self.cfg, {})
        self.assertTrue(pd.isna(frame.loc[0, 'classe_risco_desperdicio']))
        self.assertTrue(pd.isna(frame.loc[0, 'gpu_ociosa']))

    def test_sobreuso_preservado_e_negativo_invalidado(self) -> None:
        raw = fixture()
        raw.loc[0, 'gpu_wrk_util'] = 180
        raw.loc[1, 'avg_mem'] = -1
        frame = atributos(raw, self.cfg, {})
        self.assertEqual(frame.loc[0, 'utilizacao_gpu'], 180)
        self.assertTrue(frame.loc[0, 'utilizacao_gpu_fora_intervalo'])
        self.assertTrue(pd.isna(frame.loc[1, 'classe_risco_desperdicio']))

    def test_cardinalidade_conflitante_nao_escolhe_linha(self) -> None:
        data = pd.DataFrame({'id':[1,1], 'valor':[2,3]})
        with self.assertRaises(ValueError):
            dimensao_unica(data, ['id'], 'fixture', {})
        clean = dimensao_unica(pd.DataFrame({'id':[1,1], 'valor':[2,2]}), ['id'], 'fixture', {})
        self.assertEqual(len(clean), 1)

    def test_temporal_sem_jobs_compartilhados(self) -> None:
        frame = dividir(self.df, self.cfg, {})
        subsets = [frame[frame.conjunto.eq(s)] for s in ['treino','validacao','teste']]
        for a, b in zip(subsets, subsets[1:]):
            self.assertLess(a.fim.max(), b.inicio.min())
            self.assertFalse(set(a.inst_id) & set(b.inst_id))

    def test_estratificacao_e_servidores(self) -> None:
        for mode in ['estratificado', 'servidores']:
            cfg = {**self.cfg, 'split': mode}
            frame = dividir(self.df, cfg, {})
            sets = [set(frame.loc[frame.conjunto.eq(s), 'inst_id']) for s in ['treino','validacao','teste']]
            for i in range(3):
                for j in range(i+1,3):
                    self.assertFalse(sets[i] & sets[j])
            if mode == 'servidores':
                sets = [set(frame.loc[frame.conjunto.eq(s), 'machine']) for s in ['treino','validacao','teste']]
                self.assertFalse(sets[0] & sets[2])

    def test_teste_nao_altera_imputacao_escala_outliers(self) -> None:
        frame = dividir(self.df, self.cfg, {})
        train = frame[frame.conjunto.eq('treino')]
        report = {}
        prep, cols = montar_preprocessador(train, self.cfg, report)
        self.assertFalse(any('utilizacao' in c or 'score' in c or c in IDENTIDADE for c in cols if c != 'task_name'))
        x = frame[cols].astype(object).where(frame[cols].notna(), np.nan)
        prep.fit(x.loc[train.index], train.classe_risco_desperdicio)
        before = prep.transform(x.loc[train.index])
        testmask = frame.conjunto.eq('teste')
        frame.loc[testmask, 'recursos_cpu_alocados'] = 1e12
        x.loc[testmask, 'recursos_cpu_alocados'] = 1e12
        np.testing.assert_array_equal(before, prep.transform(x.loc[train.index]))
        r1, r2 = {}, {}
        sinalizar_outliers(frame, self.cfg, r1)
        frame.loc[testmask, 'utilizacao_gpu'] = 1e15
        sinalizar_outliers(frame, self.cfg, r2)
        self.assertEqual(r1['outliers']['limiares_treino'], r2['outliers']['limiares_treino'])

    def test_config_invalida(self) -> None:
        with self.assertRaises(ValueError):
            validar_config({**self.cfg, 'proporcoes':[.7,.2,.2]})

    def test_relatorio_json_compativel_com_leitores_estritos(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'relatorio.json'
            salvar_json(path, {'valor_ausente':np.nan, 'infinito':np.inf, 'contador':np.int64(2)})
            def recusar(value: str) -> None:
                raise AssertionError(f'Constante JSON inválida: {value}')
            parsed = json.loads(path.read_text(), parse_constant=recusar)
            self.assertIsNone(parsed['valor_ausente'])
            self.assertEqual(parsed['contador'], 2)

    def test_seletores_opcionais_exportam_matrizes(self) -> None:
        frame = dividir(self.df, self.cfg, {})
        for method in ['kbest','forest','pca']:
            with tempfile.TemporaryDirectory() as temp:
                cfg = {**self.cfg, 'selecao':method, 'treinar_baseline':False, 'k_best':3}
                report = {}
                treinar(frame, cfg, Path(temp), report)
                matrix = np.load(Path(temp)/'treino_transformado.npz')['X']
                self.assertGreater(matrix.shape[1], 0)
                self.assertEqual(matrix.shape[1], len(report['selecao']['atributos_transformados']))

    def test_quarentena_nao_escolhe_intervalo_arbitrario(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            raw = pd.DataFrame({'id':['a','a','b'], 'start_time':[1.,2.,3.]})
            report = {}
            clean = resolver_ambiguidades(raw, ['id'], 'instancias', report, Path(temp))
            self.assertEqual(len(clean), 2)
            self.assertTrue(clean.loc[clean.id.eq('a'), 'start_time'].isna().all())
            self.assertEqual(len(pd.read_parquet(Path(temp)/'instancias.parquet')), 2)

    def test_fluxo_completo_csv_parquet_e_modelo(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'dados'; data.mkdir()
            raw = fixture()
            raw['plan_cpu'] = 400. + ((np.arange(len(raw))//2) % 3) * 100
            frames = {
                'pai_instance_table': raw,
                'pai_sensor_table': raw.assign(gpu_name='/dev/nvidia0'),
                'pai_task_table': raw.assign(inst_num=2.),
                'pai_machine_spec': raw,
                'pai_group_tag_table': raw.assign(user='user', group='group', gpu_type_spec='V100'),
                'pai_job_table': raw.assign(user='user'),
                'pai_machine_metric': raw,
            }
            for table, frame in frames.items():
                cols = (SCHEMA/'data'/f'{table}.header').read_text().strip().split(',')
                frame = frame.copy()
                for col in cols:
                    if col not in frame:
                        frame[col] = 0.
                selected = frame[cols].drop_duplicates()
                if table == 'pai_instance_table':
                    incomplete = selected.iloc[[0]].copy()
                    incomplete['worker_name'] = 'worker_sem_maquina'
                    incomplete['machine'] = pd.NA
                    selected = pd.concat([selected, incomplete], ignore_index=True)
                selected.to_csv(data/f'{table}.csv', index=False, header=False)
            cfg = {**self.cfg, 'fracao_amostra':1., 'chunksize':40}
            report = executar(data, root/'saida', cfg)
            csv = pd.read_csv(root/'saida/base_processada.csv')
            parquet = pd.read_parquet(root/'saida/base_processada.parquet')
            self.assertEqual(len(csv), len(parquet))
            self.assertEqual(len(parquet), len(raw))
            self.assertEqual(len(pd.read_parquet(root/'saida/quarentena/instancias_chaves_nulas.parquet')), 1)
            self.assertEqual(report['final']['duplicatas_finais_identidade'], 0)
            self.assertTrue((root/'saida/modelo_baseline.joblib').exists())
            self.assertTrue((root/'saida/relatorio_qualidade.md').exists())
            self.assertGreater(len(list((root/'saida/graficos').glob('*.png'))), 5)
            # Segunda execução verifica o cache sem recontar/alterar as fontes.
            cfg2 = {**cfg, 'janela':'15min'}
            again = executar(data, root/'saida_cache', cfg2)
            self.assertEqual(report['fontes'], again['fontes'])
            self.assertEqual(again['configuracao']['janela'], '15min')

if __name__ == '__main__':
    unittest.main(verbosity=2)

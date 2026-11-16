"""Download oficial retomável por HTTP Range e verificado por SHA-256."""
from __future__ import annotations
import hashlib
import logging
import re
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEMA = ROOT.parent / 'clusterdata/cluster-trace-gpu-v2020'
URL = 'https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/'


def download_all(destination: Path) -> None:
    """Baixa sete tabelas, retoma blocos e verifica checksums do README oficial."""
    destination.mkdir(parents=True, exist_ok=True)
    checks = re.findall(r'([0-9a-f]{64})\s+(pai_\w+\.tar\.gz)', (SCHEMA / 'README.md').read_text())
    if len(checks) != 7:
        raise ValueError('Não foram encontrados os sete checksums oficiais.')

    def digest(path: Path) -> str:
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    def fetch(item: tuple[str, str]) -> None:
        expected, name = item
        target = destination / name
        if target.exists() and digest(target) == expected:
            print(f'Verificado: {name}', flush=True)
            return
        print(f'Baixando: {name}', flush=True)
        with urllib.request.urlopen(urllib.request.Request(URL + name, method='HEAD'), timeout=60) as r:
            size = int(r.headers['Content-Length'])
        # Arquivos antigos .partial são prefixos contíguos da implementação inicial.
        prefix = target.with_suffix('.partial')
        prefix_size = prefix.stat().st_size if prefix.exists() else 0
        if prefix_size > size:
            raise ValueError(f'Prefixo maior que objeto remoto: {prefix}')
        pieces = []
        step = 8 * 1024 * 1024
        for start in range(prefix_size, size, step):
            end = min(start + step, size) - 1
            pieces.append((start, end, destination / f'{name}.range-{start}-{end}'))

        def block(spec: tuple[int, int, Path]) -> None:
            start, end, path = spec
            for attempt in range(4):
                existing = path.stat().st_size if path.exists() else 0
                if existing == end - start + 1:
                    return
                if existing > end - start + 1:
                    path.unlink(); existing = 0
                req = urllib.request.Request(URL + name, headers={'Range': f'bytes={start+existing}-{end}'})
                try:
                    with urllib.request.urlopen(req, timeout=60) as response:
                        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start+existing}-{end}/{size}':
                            raise ValueError('Servidor não respeitou HTTP Range.')
                        with path.open('ab') as out:
                            shutil.copyfileobj(response, out, length=256 * 1024)
                    if path.stat().st_size != end - start + 1:
                        raise OSError('Bloco incompleto.')
                    return
                except (OSError, TimeoutError):
                    if attempt == 3:
                        raise
                    time.sleep(1 + attempt)

        with ThreadPoolExecutor(max_workers=16) as executor:
            list(executor.map(block, pieces))
        assembled = destination / f'{name}.assembled'
        with assembled.open('wb') as out:
            if prefix_size:
                with prefix.open('rb') as source:
                    shutil.copyfileobj(source, out)
            for _, _, path in pieces:
                with path.open('rb') as source:
                    shutil.copyfileobj(source, out)
        if digest(assembled) != expected:
            assembled.unlink()
            raise ValueError(f'SHA-256 incorreto: {name}; remova prefixos/blocos e repita.')
        assembled.replace(target)
        prefix.unlink(missing_ok=True)
        for _, _, path in pieces:
            path.unlink()
        print(f'Verificado: {name}', flush=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(fetch, checks))

if __name__ == '__main__':
    download_all(ROOT / 'dados')

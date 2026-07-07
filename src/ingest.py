"""Script de ingestão a partir de uma tabela Supabase para camada bronze.

Funcionalidades:
 - Lê variáveis de ambiente: SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_TABLE
 - (Opcional) Carrega .env se existir
 - Extrai todos os registos da tabela via paginação
 - Salva em CSV e Parquet para data/bronze/<tabela>_<YYYYMMDD>.{csv,parquet}
 - Evita sobre-escrever ficheiros existentes (acrescenta sufixo incremental se necessário)
 - Logging simples no stdout

Requisitos:
  pip install supabase python-dotenv pandas pyarrow
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd

try:
	from supabase import create_client, Client
except ImportError:  # pragma: no cover
	print("[ERRO] Biblioteca 'supabase' não instalada. Adicione-a ao pyproject.toml e instale.", file=sys.stderr)
	raise

try:
	from dotenv import load_dotenv
except ImportError:  # pragma: no cover
	load_dotenv = None  # type: ignore


PAGE_SIZE = 1000  # Ajustável conforme limites / performance
MAX_PAGES = 10_000  # salvaguarda


@dataclass
class IngestConfig:
	url: str
	anon_key: str
	table: str
	page_size: int = PAGE_SIZE


def log(msg: str) -> None:
	ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
	print(f"[{ts} UTC] {msg}")


def load_config() -> IngestConfig:
	# Carrega .env se existir
	if load_dotenv is not None and os.path.isfile('.env'):
		load_dotenv(override=False)

	missing = []
	url = os.getenv('SUPABASE_URL') or ''
	if not url:
		missing.append('SUPABASE_URL')
	anon = os.getenv('SUPABASE_ANON_KEY') or ''
	if not anon:
		missing.append('SUPABASE_ANON_KEY')
	table = os.getenv('SUPABASE_TABLE') or ''
	if not table:
		missing.append('SUPABASE_TABLE')

	if missing:
		raise RuntimeError(f"Variáveis de ambiente em falta: {', '.join(missing)}")

	return IngestConfig(url=url, anon_key=anon, table=table)


def supabase_client(cfg: IngestConfig) -> Client:
	return create_client(cfg.url, cfg.anon_key)


def fetch_all_rows(client: Client, table: str, page_size: int) -> List[Dict[str, Any]]:
	"""Paginação simples usando range().

	Nota: O cliente supabase python usa postgrest. range(from_, to_) é inclusivo.
	"""
	all_rows: List[Dict[str, Any]] = []
	page = 0
	while True:
		start = page * page_size
		end = start + page_size - 1
		log(f"A buscar registos {start}-{end} ...")
		query = client.table(table).select('*').range(start, end)
		data = query.execute().data  # type: ignore[attr-defined]
		if not data:
			break
		all_rows.extend(data)
		page += 1
		if len(data) < page_size:
			# Última página
			break
		if page >= MAX_PAGES:
			raise RuntimeError("Número máximo de páginas excedido - possível loop infinito")
		# Respeitar limites de taxa (ajustar se necessário)
		time.sleep(0.2)
	log(f"Total de registos obtidos: {len(all_rows)}")
	return all_rows


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def unique_output_path(base_dir: str, table: str, date_str: str, ext: str) -> str:
	base = os.path.join(base_dir, f"{table}_{date_str}")
	candidate = f"{base}.{ext}"
	i = 1
	while os.path.exists(candidate):
		candidate = f"{base}_{i}.{ext}"
		i += 1
	return candidate


def save_outputs(df: pd.DataFrame, table: str) -> Dict[str, str]:
	date_str = datetime.now(timezone.utc).strftime('%Y%m%d')
	bronze_dir = os.path.join('data', 'bronze')
	ensure_dir(bronze_dir)
	csv_path = unique_output_path(bronze_dir, table, date_str, 'csv')
	parquet_path = unique_output_path(bronze_dir, table, date_str, 'parquet')
	log(f"A escrever CSV: {csv_path}")
	df.to_csv(csv_path, index=False)
	log(f"A escrever Parquet: {parquet_path}")
	df.to_parquet(parquet_path, index=False)
	return {"csv": csv_path, "parquet": parquet_path}


def run() -> int:
	try:
		cfg = load_config()
		log(f"Config: table={cfg.table} url={cfg.url}")
		client = supabase_client(cfg)
		rows = fetch_all_rows(client, cfg.table, cfg.page_size)
		if not rows:
			log("Aviso: 0 registos obtidos. Nada a gravar.")
			return 0
		df = pd.DataFrame(rows)
		outputs = save_outputs(df, cfg.table)
		log(f"Concluído. Ficheiros: {outputs}")
		return 0
	except Exception as e:  # pragma: no cover
		log(f"ERRO: {e}")
		return 1


if __name__ == '__main__':
	raise SystemExit(run())

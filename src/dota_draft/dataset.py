"""OpenDota ingestion and training dataset preparation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging
import time

import pandas as pd
import requests

from .config import DATA_DIR


# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


OPENDOTA_BASE_URL = "https://api.opendota.com/api"
REQUEST_DELAY = 0.2
MAX_WORKERS = 10

PATCH_MAPPING: Dict[int, str] = {}


def fetch_patch_mapping() -> Dict[int, str]:
	"""Fetch OpenDota patch ID to semantic version mapping."""
	global PATCH_MAPPING

	if PATCH_MAPPING:
		return PATCH_MAPPING

	try:
		response = requests.get(f"{OPENDOTA_BASE_URL}/constants/patch", timeout=10)
		response.raise_for_status()
		patches = response.json()

		if isinstance(patches, list):
			PATCH_MAPPING = {p["id"]: p["name"] for p in patches if "id" in p and "name" in p}
			if PATCH_MAPPING:
				latest_id = max(PATCH_MAPPING.keys())
				logger.info(
					"Loaded %s patch mappings. Latest: ID %s = %s",
					len(PATCH_MAPPING),
					latest_id,
					PATCH_MAPPING[latest_id],
				)

		return PATCH_MAPPING
	except Exception as exc:  # noqa: BLE001
		logger.warning("Tidak dapat mengambil patch mapping: %s", exc)
		return {
			52: "7.33",
			53: "7.34",
			54: "7.35",
			55: "7.36",
			56: "7.37",
			57: "7.38",
			58: "7.39",
			59: "7.40",
		}


def convert_patch_to_version(patch_id: int) -> str:
	"""Convert OpenDota patch ID to semantic patch string."""
	if not PATCH_MAPPING:
		fetch_patch_mapping()
	return PATCH_MAPPING.get(patch_id, str(patch_id))


def get_weight(match_patch: str, current_patch: str) -> float:
	"""Compute instance weight based on patch distance."""
	try:
		match_version = int(match_patch)
		current_version = int(current_patch)
		version_diff = abs(current_version - match_version)
		if version_diff == 0:
			return 1.0
		if version_diff == 1:
			return 0.5
		return 0.1
	except (ValueError, TypeError):
		try:
			match_version = float(match_patch)
			current_version = float(current_patch)
			version_diff = abs(current_version - match_version)
			if version_diff < 0.01:
				return 1.0
			if version_diff < 0.02:
				return 0.5
			return 0.1
		except (ValueError, TypeError):
			return 0.1


def fetch_public_matches(min_rank_tier: int = 75) -> List[Dict]:
	"""Fetch public matches from OpenDota with rank tier filtering."""
	url = f"{OPENDOTA_BASE_URL}/publicMatches"
	try:
		params = {"min_rank": min_rank_tier} if min_rank_tier > 0 else {}
		response = requests.get(url, params=params, timeout=10)
		response.raise_for_status()
		return response.json()
	except requests.exceptions.RequestException as exc:
		logger.error("Error mengambil public matches: %s", exc)
		return []


def fetch_pro_matches(less_than_match_id: Optional[int] = None) -> List[Dict]:
	"""Fetch pro matches from OpenDota with optional pagination."""
	url = f"{OPENDOTA_BASE_URL}/proMatches"
	params = {}
	if less_than_match_id:
		params["less_than_match_id"] = less_than_match_id

	try:
		response = requests.get(url, params=params, timeout=10)
		response.raise_for_status()
		return response.json()
	except requests.exceptions.RequestException as exc:
		logger.error("Error mengambil pro matches: %s", exc)
		return []


def fetch_match_details(match_id: int, max_retries: int = 3) -> Optional[Dict]:
	"""Fetch match details with retry and exponential backoff on rate limit."""
	url = f"{OPENDOTA_BASE_URL}/matches/{match_id}"

	for attempt in range(max_retries):
		try:
			response = requests.get(url, timeout=10)
			if response.status_code == 429:
				wait_time = (2**attempt) * 0.5
				logger.debug(
					"Rate limited untuk match %s, menunggu %ss sebelum retry %s/%s",
					match_id,
					wait_time,
					attempt + 1,
					max_retries,
				)
				time.sleep(wait_time)
				continue

			response.raise_for_status()
			return response.json()
		except requests.exceptions.RequestException as exc:
			if attempt == max_retries - 1:
				logger.warning(
					"Error mengambil match %s setelah %s attempts: %s",
					match_id,
					max_retries,
					exc,
				)
				return None
			time.sleep(0.5)

	return None


def parse_match_data(match_detail: Dict) -> Optional[Dict]:
	"""Parse raw OpenDota match detail into project training format."""
	try:
		radiant_heroes: List[int] = []
		dire_heroes: List[int] = []

		players = match_detail.get("players", [])
		if len(players) != 10:
			return None

		for player in players:
			hero_id = player.get("hero_id")
			if not hero_id:
				return None

			if player.get("isRadiant"):
				radiant_heroes.append(hero_id)
			else:
				dire_heroes.append(hero_id)

		if len(radiant_heroes) != 5 or len(dire_heroes) != 5:
			return None

		patch_id = match_detail.get("patch")
		patch_version = convert_patch_to_version(patch_id) if isinstance(patch_id, int) else str(patch_id)

		start_time = match_detail.get("start_time", 0)
		match_date = (
			datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S") if start_time else None
		)

		series_id = match_detail.get("series_id", 0)
		avg_mmr = match_detail.get("average_mmr", 0)

		match = {
			"match_id": match_detail.get("match_id"),
			"start_time": start_time,
			"match_date": match_date,
			"patch": patch_version,
			"patch_id": patch_id,
			"radiant_team": radiant_heroes,
			"dire_team": dire_heroes,
			"radiant_win": 1 if match_detail.get("radiant_win") else 0,
			"duration": match_detail.get("duration", 0),
		}

		if series_id:
			match["match_type"] = "pro"
			match["series_id"] = series_id
		else:
			match["match_type"] = "public"
			match["avg_mmr"] = avg_mmr

		return match
	except Exception as exc:  # noqa: BLE001
		logger.warning("Error parsing match data: %s", exc)
		return None


def fetch_opendota_matches(
	num_pro_matches: int = 100,
	num_public_matches: int = 100,
	latest_patch_only: bool = False,
) -> List[Dict]:
	"""Fetch pro and public matches and return parsed training-ready records."""
	all_matches: List[Dict] = []

	logger.info("Mengambil professional matches untuk data MMR tinggi...")
	logger.info("Target: %s pro matches (fetching dalam batch ~100)", num_pro_matches)

	pro_matches: List[Dict] = []
	less_than_id = None
	batches_needed = (num_pro_matches // 100) + 1

	for _ in range(1, batches_needed + 1):
		batch = fetch_pro_matches(less_than_match_id=less_than_id)
		if not batch:
			logger.warning("Tidak ada pro matches lagi setelah %s matches", len(pro_matches))
			break

		pro_matches.extend(batch)
		less_than_id = batch[-1].get("match_id")

		if len(pro_matches) >= num_pro_matches * 2:
			break

		time.sleep(0.1)

	logger.info("Berhasil mengambil %s professional match listings", len(pro_matches))
	all_matches.extend(pro_matches[: num_pro_matches * 2])

	logger.info("Mengambil public matches dengan rank tier 75+ (Divine 5)...")
	public_matches = fetch_public_matches(min_rank_tier=75)
	logger.info(
		"Berhasil mengambil %s public match listings (rank tier >= 75)",
		len(public_matches),
	)
	all_matches.extend(public_matches[: num_public_matches * 2])

	if not all_matches:
		logger.error("Tidak ada matches yang ditemukan dari OpenDota API.")
		return []

	latest_patch = None
	if latest_patch_only:
		logger.info("Filtering untuk patch terbaru saja...")

	matches: List[Dict] = []
	target_total = num_pro_matches + num_public_matches
	target_count = len(all_matches)

	logger.info(
		"Mengambil data detail untuk hingga %s matches (menggunakan %s concurrent workers)...",
		target_count,
		MAX_WORKERS,
	)
	logger.info("Estimasi waktu: ~%.1f menit", (target_count * REQUEST_DELAY / MAX_WORKERS) / 60)

	def fetch_and_parse_match(match_listing: Dict) -> Optional[Dict]:
		match_id = match_listing.get("match_id")
		time.sleep(REQUEST_DELAY)
		match_detail = fetch_match_details(match_id)
		if match_detail:
			return parse_match_data(match_detail)
		return None

	match_ids_to_fetch = all_matches[:target_count]

	with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
		futures = [executor.submit(fetch_and_parse_match, match) for match in match_ids_to_fetch]
		for future in as_completed(futures):
			try:
				parsed_match = future.result()
				if not parsed_match:
					continue

				if latest_patch_only and latest_patch is None:
					latest_patch = parsed_match["patch"]
					logger.info("Patch terbaru terdeteksi: %s", latest_patch)

				if latest_patch_only and parsed_match["patch"] != latest_patch:
					continue

				matches.append(parsed_match)

				if len(matches) % 50 == 0 or len(matches) >= target_total:
					if latest_patch_only:
						logger.info(
							"Progress: %s/%s matches berhasil diambil (Patch %s)",
							len(matches),
							target_total,
							latest_patch,
						)
					else:
						logger.info(
							"Progress: %s/%s matches berhasil diambil",
							len(matches),
							target_total,
						)

				if len(matches) >= target_total:
					break
			except Exception as exc:  # noqa: BLE001
				logger.warning("Error memproses match: %s", exc)

	logger.info("Berhasil mengambil %s matches dari OpenDota API.", len(matches))
	return matches


def prepare_training_dataframe(matches: List[Dict], current_patch: Optional[str] = None) -> pd.DataFrame:
	"""Convert match list to weighted training DataFrame."""
	logger.info("Menyiapkan training DataFrame dengan weighted instances...")

	if current_patch is None and matches:
		try:
			patch_ids = [int(m["patch"]) for m in matches if m.get("patch")]
			current_patch = str(max(patch_ids))
			logger.info("Auto-detected patch saat ini: %s", current_patch)
		except (ValueError, TypeError):
			current_patch = "7.40"

	rows = []
	for match in matches:
		row = {
			"match_id": match["match_id"],
			"match_date": match.get("match_date"),
			"start_time": match.get("start_time", 0),
			"patch": match["patch"],
			"duration": match["duration"],
			"radiant_win": match["radiant_win"],
			"match_type": match.get("match_type", "unknown"),
		}

		if match.get("match_type") == "pro":
			row["series_id"] = match.get("series_id", 0)
		elif match.get("match_type") == "public":
			row["avg_mmr"] = match.get("avg_mmr", 0)

		for i, hero_id in enumerate(match["radiant_team"], 1):
			row[f"radiant_hero_{i}"] = hero_id

		for i, hero_id in enumerate(match["dire_team"], 1):
			row[f"dire_hero_{i}"] = hero_id

		row["weight"] = get_weight(str(match["patch"]), str(current_patch))
		rows.append(row)

	df = pd.DataFrame(rows)
	if df.empty:
		return df

	weight_counts = df["weight"].value_counts().sort_index(ascending=False)
	logger.info("Distribusi weight:")
	for weight, count in weight_counts.items():
		logger.info("  Weight %s: %s matches (%.1f%%)", weight, count, count / len(df) * 100)

	if "match_type" in df.columns:
		match_type_counts = df["match_type"].value_counts()
		logger.info("Distribusi tipe match:")
		for match_type, count in match_type_counts.items():
			logger.info("  %s: %s matches (%.1f%%)", match_type.capitalize(), count, count / len(df) * 100)

	return df


def run_ingestion(
	num_pro_matches: int = 100,
	num_public_matches: int = 100,
	latest_patch_only: bool = True,
	output_path: Optional[Path] = None,
) -> pd.DataFrame:
	"""Execute end-to-end ingestion and save weighted training dataset."""
	logger.info("=" * 60)
	logger.info("MetaMorph - Zone A: Ingestion & Patch Watcher")
	logger.info("=" * 60)

	fetch_patch_mapping()
	matches = fetch_opendota_matches(
		num_pro_matches=num_pro_matches,
		num_public_matches=num_public_matches,
		latest_patch_only=latest_patch_only,
	)
	df = prepare_training_dataframe(matches, current_patch=None)

	if output_path is None:
		output_path = DATA_DIR / "processed" / "training_data_weighted.csv"

	output_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(output_path, index=False)
	logger.info("Training data disimpan ke: %s", output_path)

	return df


def main() -> None:
	"""CLI-like entrypoint for ingestion module."""
	df = run_ingestion(num_pro_matches=100, num_public_matches=100, latest_patch_only=True)
	if df.empty:
		logger.warning("DataFrame kosong. Cek koneksi/API limit OpenDota.")
		return

	logger.info("Preview DataFrame (5 baris pertama):")
	print(df.head())
	logger.info("Bentuk DataFrame: %s", df.shape)
	logger.info("Kolom: %s", list(df.columns))


if __name__ == "__main__":
	main()


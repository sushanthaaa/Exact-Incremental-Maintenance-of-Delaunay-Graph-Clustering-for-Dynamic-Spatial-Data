#!/usr/bin/env python3
"""Generate synthetic and real 2D spatial datasets for DelauCluster."""

from __future__ import annotations

import argparse
import io
import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_circles, make_moons


EARTH_RADIUS_KM = 6371.0088

SYNTHETIC_DATASETS = [
    "moons",
    "circles",
    "spiral",
    "varying_density",
    "touching",
    "chain_noise",
    "anisotropic",
    "uniform_noise",
]
DATASETS = SYNTHETIC_DATASETS


@dataclass(frozen=True)
class RealDatasetSpec:
    name: str
    source_name: str
    source_url: str
    source_page: str
    lon_column: str
    lat_column: str
    label_column: str | None = None
    notes: str = ""
    paged_params: dict[str, str] | None = None
    page_size: int = 0


def save_csv(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": x[:, 0], "y": x[:, 1], "label": y.astype(int)}).to_csv(
        path, index=False, header=False
    )
    print(f"saved {path} ({len(x)} points)")


def socrata_url(domain: str, dataset_id: str, params: dict[str, str]) -> str:
    return f"https://{domain}/resource/{dataset_id}.csv?{urllib.parse.urlencode(params)}"


def real_dataset_specs(limit: int) -> dict[str, RealDatasetSpec]:
    usgs_params = {
        "format": "csv",
        "starttime": "2024-01-01",
        "endtime": "2024-12-31",
        "minmagnitude": "1.0",
        "minlatitude": "32",
        "maxlatitude": "42",
        "minlongitude": "-125",
        "maxlongitude": "-114",
        "eventtype": "earthquake",
        "orderby": "time-asc",
        "limit": str(max(limit, 1)),
    }
    sf_params = {
        "$select": "longitude,latitude,incident_category",
        "$where": (
            "latitude IS NOT NULL AND longitude IS NOT NULL "
            "AND incident_datetime between '2025-01-01T00:00:00' "
            "and '2025-12-31T23:59:59'"
        ),
        "$order": "incident_datetime",
        "$limit": str(max(limit, 1)),
    }
    nyc_params = {
        "$select": "longitude,latitude,borough",
        "$where": (
            "latitude IS NOT NULL AND longitude IS NOT NULL "
            "AND latitude != 0 AND longitude != 0 "
            "AND crash_date between '2025-01-01T00:00:00' "
            "and '2025-12-31T23:59:59'"
        ),
        "$order": "crash_date",
        "$limit": str(max(limit, 1)),
    }
    usgs_global_params = {
        "format": "csv",
        "starttime": "2010-01-01",
        "endtime": "2026-05-20",
        "minmagnitude": "2.5",
        "eventtype": "earthquake",
        "orderby": "time-asc",
    }
    usgs_western_us_params = {
        "format": "csv",
        "starttime": "1980-01-01",
        "endtime": "2026-05-20",
        "minmagnitude": "1.0",
        "minlatitude": "31",
        "maxlatitude": "49",
        "minlongitude": "-125",
        "maxlongitude": "-102",
        "eventtype": "earthquake",
        "orderby": "time-asc",
    }
    nyc_large_params = {
        "$select": "longitude,latitude,borough",
        "$where": (
            "latitude IS NOT NULL AND longitude IS NOT NULL "
            "AND latitude != 0 AND longitude != 0 "
            "AND crash_date between '2012-07-01T00:00:00' "
            "and '2026-05-20T23:59:59'"
        ),
        "$order": "crash_date,collision_id",
        "$limit": str(max(limit, 1)),
    }
    return {
        "usgs_global_earthquakes_2010_2026_large": RealDatasetSpec(
            name="usgs_global_earthquakes_2010_2026_large",
            source_name="USGS Earthquake Catalog API",
            source_url=(
                "https://earthquake.usgs.gov/fdsnws/event/1/query.csv?"
                + urllib.parse.urlencode({**usgs_global_params, "limit": min(max(limit, 1), 20000)})
            ),
            source_page="https://earthquake.usgs.gov/fdsnws/event/1/",
            lon_column="longitude",
            lat_column="latitude",
            notes=(
                "Global USGS earthquake epicenters, magnitude >= 2.5, "
                "2010-01-01 through 2026-05-20. Fetched chronologically with "
                "USGS limit/offset pagination."
            ),
            paged_params=usgs_global_params,
            page_size=10000,
        ),
        "usgs_western_us_earthquakes_1980_2026_large": RealDatasetSpec(
            name="usgs_western_us_earthquakes_1980_2026_large",
            source_name="USGS Earthquake Catalog API",
            source_url=(
                "https://earthquake.usgs.gov/fdsnws/event/1/query.csv?"
                + urllib.parse.urlencode(
                    {**usgs_western_us_params, "limit": min(max(limit, 1), 20000)}
                )
            ),
            source_page="https://earthquake.usgs.gov/fdsnws/event/1/",
            lon_column="longitude",
            lat_column="latitude",
            notes=(
                "Western US USGS earthquake epicenters, magnitude >= 1.0, "
                "1980-01-01 through 2026-05-20, fetched chronologically with "
                "USGS limit/offset pagination."
            ),
            paged_params=usgs_western_us_params,
            page_size=10000,
        ),
        "usgs_california_earthquakes_2024": RealDatasetSpec(
            name="usgs_california_earthquakes_2024",
            source_name="USGS Earthquake Catalog API",
            source_url=(
                "https://earthquake.usgs.gov/fdsnws/event/1/query.csv?"
                + urllib.parse.urlencode(usgs_params)
            ),
            source_page="https://earthquake.usgs.gov/fdsnws/event/1/",
            lon_column="longitude",
            lat_column="latitude",
            notes="California/Nevada regional earthquake epicenters, magnitude >= 1.0, 2024.",
        ),
        "sf_incidents_2025": RealDatasetSpec(
            name="sf_incidents_2025",
            source_name="DataSF Police Department Incident Reports: 2018 to Present",
            source_url=socrata_url("data.sfgov.org", "wg3w-h783", sf_params),
            source_page="https://data.sfgov.org/d/wg3w-h783",
            lon_column="longitude",
            lat_column="latitude",
            label_column="incident_category",
            notes="Incident locations are anonymized by DataSF to nearby intersections.",
        ),
        "nyc_collisions_2025": RealDatasetSpec(
            name="nyc_collisions_2025",
            source_name="NYC Open Data Motor Vehicle Collisions - Crashes",
            source_url=socrata_url("data.cityofnewyork.us", "h9gi-nx95", nyc_params),
            source_page="https://data.cityofnewyork.us/d/h9gi-nx95",
            lon_column="longitude",
            lat_column="latitude",
            label_column="borough",
            notes="Police-reported NYC vehicle collision locations for 2025.",
        ),
        "nyc_collisions_2012_2026_large": RealDatasetSpec(
            name="nyc_collisions_2012_2026_large",
            source_name="NYC Open Data Motor Vehicle Collisions - Crashes",
            source_url=socrata_url("data.cityofnewyork.us", "h9gi-nx95", nyc_large_params),
            source_page="https://data.cityofnewyork.us/d/h9gi-nx95",
            lon_column="longitude",
            lat_column="latitude",
            label_column="borough",
            notes=(
                "Police-reported NYC vehicle collision locations from the public "
                "NYC Open Data crash table, ordered by crash date and collision id."
            ),
        ),
    }


def parse_real_dataset_names(value: str) -> list[str]:
    if value == "all":
        return list(real_dataset_specs(1))
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_local_csv_specs(values: list[str]) -> list[RealDatasetSpec]:
    specs: list[RealDatasetSpec] = []
    for value in values:
        if "=" not in value:
            raise ValueError(
                "--local-csv entries must look like name=/path/file.csv,lon,lat[,label]"
            )
        name, rest = value.split("=", 1)
        parts = [part.strip() for part in rest.split(",") if part.strip()]
        if len(parts) < 3:
            raise ValueError(
                "--local-csv entries must include path, longitude column, and latitude column"
            )
        source_path = Path(parts[0]).expanduser()
        specs.append(
            RealDatasetSpec(
                name=name.strip(),
                source_name="local CSV",
                source_url=source_path.resolve().as_uri(),
                source_page=source_path.resolve().as_uri(),
                lon_column=parts[1],
                lat_column=parts[2],
                label_column=parts[3] if len(parts) >= 4 else None,
                notes="Local user-provided spatial CSV.",
            )
        )
    return specs


def paged_url(spec: RealDatasetSpec, limit: int, offset: int) -> str:
    if spec.paged_params is None:
        raise ValueError(f"{spec.name} is not a paged dataset")
    params = dict(spec.paged_params)
    params["limit"] = str(limit)
    params["offset"] = str(offset)
    return "https://earthquake.usgs.gov/fdsnws/event/1/query.csv?" + urllib.parse.urlencode(params)


def read_remote_csv(url: str, timeout: int, retries: int = 3) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                data = response.read()
            return pd.read_csv(io.BytesIO(data), low_memory=False)
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= retries:
                break
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}") from last_error


def download_csv(
    spec: RealDatasetSpec,
    raw_path: Path,
    timeout: int,
    force: bool,
    max_rows: int,
) -> None:
    if raw_path.exists() and not force:
        return
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.paged_params is None:
        with urllib.request.urlopen(spec.source_url, timeout=timeout) as response:
            raw_path.write_bytes(response.read())
        return

    target_rows = max_rows if max_rows > 0 else spec.page_size
    page_size = max(1, spec.page_size)
    offset = 1
    frames: list[pd.DataFrame] = []
    fetched = 0
    while fetched < target_rows:
        limit = min(page_size, target_rows - fetched)
        url = paged_url(spec, limit, offset)
        print(f"  page offset={offset} limit={limit}", flush=True)
        frame = read_remote_csv(url, timeout)
        if frame.empty:
            break
        frames.append(frame)
        rows = len(frame)
        fetched += rows
        if rows < limit:
            break
        offset += rows
    if not frames:
        raise RuntimeError(f"no rows fetched for {spec.name}")
    pd.concat(frames, ignore_index=True).to_csv(raw_path, index=False)


def equirectangular_km(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    lon0 = float(np.mean(lon))
    lat0 = float(np.mean(lat))
    x = EARTH_RADIUS_KM * np.deg2rad(lon - lon0) * math.cos(math.radians(lat0))
    y = EARTH_RADIUS_KM * np.deg2rad(lat - lat0)
    return x, y, lon0, lat0


def normalize_real_dataset(
    spec: RealDatasetSpec,
    raw_path: Path,
    output_path: Path,
    metadata_path: Path,
    max_points: int,
    seed: int,
    include_labels: bool,
) -> dict:
    raw = pd.read_csv(raw_path, low_memory=False)
    if spec.lon_column not in raw.columns or spec.lat_column not in raw.columns:
        raise ValueError(
            f"{spec.name}: missing coordinate columns {spec.lon_column!r}/{spec.lat_column!r}"
        )
    lon = pd.to_numeric(raw[spec.lon_column], errors="coerce")
    lat = pd.to_numeric(raw[spec.lat_column], errors="coerce")
    valid = lon.between(-180, 180) & lat.between(-90, 90)
    valid &= lon.notna() & lat.notna()
    frame = raw.loc[valid].copy()
    frame["_lon"] = lon.loc[valid].to_numpy(float)
    frame["_lat"] = lat.loc[valid].to_numpy(float)
    frame = frame.drop_duplicates(subset=["_lon", "_lat"])

    if max_points > 0 and len(frame) > max_points:
        frame = frame.sample(n=max_points, random_state=seed).sort_index()

    x, y, lon0, lat0 = equirectangular_km(
        frame["_lon"].to_numpy(float), frame["_lat"].to_numpy(float)
    )
    output = pd.DataFrame({"x": x, "y": y})
    label_count = 0
    if include_labels and spec.label_column and spec.label_column in frame.columns:
        labels = frame[spec.label_column].fillna("unknown").astype(str)
        output["label"] = pd.factorize(labels, sort=True)[0].astype(int)
        label_count = int(output["label"].nunique())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    metadata = {
        "dataset": spec.name,
        "source_name": spec.source_name,
        "source_url": spec.source_url,
        "source_page": spec.source_page,
        "raw_path": str(raw_path),
        "output_path": str(output_path),
        "raw_rows": int(len(raw)),
        "valid_coordinate_rows": int(valid.sum()),
        "deduplicated_rows": int(len(frame)),
        "output_rows": int(len(output)),
        "longitude_column": spec.lon_column,
        "latitude_column": spec.lat_column,
        "label_column": spec.label_column if include_labels else None,
        "label_count": label_count,
        "projection": "local equirectangular kilometers",
        "projection_center_longitude": lon0,
        "projection_center_latitude": lat0,
        "notes": spec.notes,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def standardize(x: np.ndarray) -> np.ndarray:
    std = x.std(axis=0)
    std[std < 1e-12] = 1.0
    return (x - x.mean(axis=0)) / std


def spiral(seed: int, n: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    half = n // 2
    theta = np.linspace(0.2, 4.0 * np.pi, half)
    r = theta
    a = np.c_[r * np.cos(theta), r * np.sin(theta)]
    b = np.c_[-r * np.cos(theta), -r * np.sin(theta)]
    x = np.vstack([a, b]) + rng.normal(0, 0.45, size=(2 * half, 2))
    y = np.r_[np.zeros(half), np.ones(half)]
    return standardize(x), y


def varying_density(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x0 = rng.normal([-2.0, 0.0], [0.18, 0.18], size=(500, 2))
    x1 = rng.normal([1.2, 0.0], [0.65, 0.65], size=(900, 2))
    x2 = rng.normal([0.0, 2.0], [0.32, 0.75], size=(500, 2))
    x = np.vstack([x0, x1, x2])
    y = np.r_[np.zeros(len(x0)), np.ones(len(x1)), np.full(len(x2), 2)]
    return standardize(x), y


def touching(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x0 = rng.normal([-0.55, 0.0], [0.42, 0.18], size=(700, 2))
    x1 = rng.normal([0.55, 0.0], [0.42, 0.18], size=(700, 2))
    bridge = rng.normal([0.0, 0.0], [0.05, 0.05], size=(40, 2))
    x = np.vstack([x0, x1, bridge])
    y = np.r_[np.zeros(len(x0)), np.ones(len(x1)), np.full(len(bridge), -1)]
    return standardize(x), y


def chain_noise(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x0 = rng.normal([-2.0, 0.0], 0.22, size=(500, 2))
    x1 = rng.normal([2.0, 0.0], 0.22, size=(500, 2))
    chain_x = np.linspace(-1.45, 1.45, 80)
    chain = np.c_[chain_x, rng.normal(0, 0.035, size=len(chain_x))]
    x = np.vstack([x0, x1, chain])
    y = np.r_[np.zeros(len(x0)), np.ones(len(x1)), np.full(len(chain), -1)]
    return standardize(x), y


def anisotropic(seed: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = make_blobs(n_samples=1500, centers=3, random_state=seed)
    transform = np.array([[0.6, -0.8], [1.7, 0.35]])
    return standardize(x @ transform), y


def uniform_noise(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x, y = make_blobs(
        n_samples=1200,
        centers=[[-2, -1], [2, -1], [0, 2]],
        cluster_std=[0.28, 0.34, 0.3],
        random_state=seed,
    )
    noise = rng.uniform(-4, 4, size=(250, 2))
    x = np.vstack([x, noise])
    y = np.r_[y, np.full(len(noise), -1)]
    return standardize(x), y


def build(name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if name == "moons":
        x, y = make_moons(n_samples=1200, noise=0.07, random_state=seed)
        return standardize(x), y
    if name == "circles":
        x, y = make_circles(n_samples=1200, noise=0.04, factor=0.42, random_state=seed)
        return standardize(x), y
    if name == "spiral":
        return spiral(seed)
    if name == "varying_density":
        return varying_density(seed)
    if name == "touching":
        return touching(seed)
    if name == "chain_noise":
        return chain_noise(seed)
    if name == "anisotropic":
        return anisotropic(seed)
    if name == "uniform_noise":
        return uniform_noise(seed)
    raise ValueError(f"unknown dataset: {name}")


def generate_synthetic(args: argparse.Namespace) -> None:
    names = (
        SYNTHETIC_DATASETS
        if args.datasets == "all"
        else [s.strip() for s in args.datasets.split(",") if s.strip()]
    )
    out_dir = Path(args.out_dir)
    for name in names:
        x, y = build(name, args.seed)
        save_csv(out_dir / f"{name}.csv", x, y)


def generate_real(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else out_dir / "raw"
    meta_dir = out_dir / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    download_limit = args.download_limit if args.download_limit > 0 else args.max_points
    builtins = real_dataset_specs(download_limit)
    specs: list[RealDatasetSpec] = []
    for name in parse_real_dataset_names(args.datasets):
        if name not in builtins:
            raise ValueError(f"unknown real dataset: {name}")
        specs.append(builtins[name])
    specs.extend(parse_local_csv_specs(args.local_csv))

    manifest_rows: list[dict] = []
    for spec in specs:
        raw_path = raw_dir / f"{spec.name}_raw.csv"
        if spec.source_url.startswith("file://"):
            raw_path = Path(urllib.parse.urlparse(spec.source_url).path)
        else:
            print(f"fetching {spec.name}", flush=True)
            download_csv(spec, raw_path, args.timeout, args.force, download_limit)
        output_path = out_dir / f"{spec.name}.csv"
        metadata_path = meta_dir / f"{spec.name}.json"
        metadata = normalize_real_dataset(
            spec,
            raw_path,
            output_path,
            metadata_path,
            args.max_points,
            args.seed,
            args.include_labels,
        )
        manifest_rows.append(metadata)
        print(f"wrote {output_path} ({metadata['output_rows']} points)", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "real_spatial_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"wrote {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--raw-dir", default="")
    parser.add_argument("--max-points", type=int, default=10000)
    parser.add_argument(
        "--download-limit",
        type=int,
        default=0,
        help="Rows to request before normalization. Defaults to --max-points.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--include-labels",
        action="store_true",
        help="Include source category labels for real datasets when available.",
    )
    parser.add_argument(
        "--local-csv",
        action="append",
        default=[],
        help="Real source as name=/path/file.csv,longitude_column,latitude_column[,label_column].",
    )
    args = parser.parse_args()

    if not args.out_dir:
        args.out_dir = "data/real_spatial" if args.source == "real" else "data/synthetic"

    if args.source == "real":
        generate_real(args)
    else:
        generate_synthetic(args)


if __name__ == "__main__":
    main()

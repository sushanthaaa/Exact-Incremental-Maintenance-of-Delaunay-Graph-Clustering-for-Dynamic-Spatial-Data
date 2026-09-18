# Paper input snapshots

Run `python scripts/restore_data.py` from the repository root. The script checks SHA-256 hashes of both compressed archives and decompressed bytes, and restores five inputs to the locations expected by the original experiment scripts. `--verify-only` checks the archives and any existing restored inputs without writing. Changed local inputs are never silently replaced. `manifest.json` is the authoritative byte-level inventory.

| Restored input | Use and source |
|---|---|
| `real_spatial_large/nyc_collisions_2012_2026_large.csv` | Projected NYC collision locations; modeled correction workload |
| `real_spatial_large/usgs_western_us_earthquakes_1980_2026_large.csv` | Projected USGS western-US locations; modeled corrections |
| `comcat_revision/socal_events.csv` | Cached southern-California ComCat events; delayed-withdrawal and frozen-threshold studies |
| `comcat_revision/socal_decluster.csv` | Cached ComCat input for the declustering workload |
| `ais_lalb/ais_lalb_big.csv` | Cached LA/Long Beach vessel reports for 2023-01-01 |

The NYC and western-US normalized files are headerless `x,y,label` files. ComCat files have column headers. AIS is headerless `mmsi,t,lat,lon,status`; vessel identifiers associate reports when constructing updates. These are research input snapshots, not live feeds.

## Sources and attribution

- NYC Open Data, [Motor Vehicle Collisions — Crashes](https://data.cityofnewyork.us/d/h9gi-nx95). See the [NYC Open Data terms](https://data.cityofnewyork.us/stories/s/Terms-of-Use/k9k7-3cje/). Projected snapshots and retained query metadata are in this release; raw API responses are not.
- USGS, [Earthquake Catalog / ComCat](https://earthquake.usgs.gov/earthquakes/search/). USGS-produced data are generally in the U.S. public domain; see [USGS data licensing](https://www.usgs.gov/data-management/data-licensing). Preserve attribution and check upstream exceptions when adding other material.
- NOAA Office for Coastal Management, BOEM, and U.S. Coast Guard, [Vessel Traffic / AIS](https://www.coast.noaa.gov/digitalcoast/data/vesseltraffic.html). The retained AIS provenance records a copy into the research project on 2026-07-11. It does not record the original upstream acquisition date or supply a complete original extraction recipe. The exact cached input is included so this experiment does not depend on reconstructing that missing acquisition history.

Third-party data are not relicensed under the repository's software GPL. These research subsets are not official agency products or endorsed by the source agencies.

## Provenance limits

`provenance/` preserves the available historical records unchanged. The normalized metadata reports 19,517 NYC and 48,877 western-US output rows, whereas counting the actual headerless snapshots gives 19,518 and 48,878. Use the files, checksums, and duplicate census for the actual experimental inputs; the original metadata discrepancy is retained explicitly rather than silently repaired.

Some scripts reuse a cache whenever it exists, even if command-line date or magnitude parameters differ. In particular, the rethreshold-boundary script's query defaults differ from the cached revision dataset. For paper reproduction, restore these snapshots and do not use `--refetch`. To study a different period, work in a separate checkout/cache and record the new input hashes. A new download may differ as upstream catalogs are revised.

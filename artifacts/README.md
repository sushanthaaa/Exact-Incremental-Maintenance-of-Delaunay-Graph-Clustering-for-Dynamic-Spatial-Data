# Archived evidence

`paper/` holds a curated set of historical experiment results, including summaries, curves, and selector inputs/traces. See [the evidence map](../docs/provenance.md). New experiment output belongs in ignored `results/`, not here.

`source_manifest.json` records the imported paths and SHA-256 hashes. `source_path` is relative to the earlier research checkout. For the smoke driver and license notice, `source_sha256` identifies the original file and `sha256` identifies the released version. The driver discovers the added snapshot tests; the notice has a trailing blank line removed. The C++ engine and original Python experiment scripts are copied unchanged.

To check imported file integrity from the repository root:

```bash
python - <<'PYCODE'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('artifacts/source_manifest.json').read_text())
for entry in manifest['files']:
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
print('Imported file hashes match')
PYCODE
```

"""Protect the integrity of released inputs and existing local data."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "restore_data", Path(__file__).resolve().parents[1] / "scripts/restore_data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "data/snapshots").mkdir(parents=True)
        self.raw = b"1.25,2.75,0\n"
        packed = gzip.compress(self.raw, mtime=0)
        (self.root / "data/snapshots/test.gz").write_bytes(packed)
        self.entry = {"archive": "data/snapshots/test.gz", "destination": "data/input.csv",
                      "sha256": hashlib.sha256(self.raw).hexdigest(),
                      "archive_sha256": hashlib.sha256(packed).hexdigest(),
                      "bytes": len(self.raw)}
        self.save_manifest()

    def save_manifest(self):
        (self.root / "data/manifest.json").write_text(json.dumps(
            {"schema_version": 1, "snapshots": [self.entry]}))

    def test_verify_restore_and_repeat(self):
        self.assertEqual(module.restore(self.root, True), 1)
        target = self.root / "data/input.csv"
        self.assertFalse(target.exists())
        module.restore(self.root)
        self.assertEqual(target.read_bytes(), self.raw)
        module.restore(self.root)

    def test_corrupt_archive_rejected(self):
        (self.root / self.entry["archive"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Archive checksum"):
            module.restore(self.root)
        self.assertFalse((self.root / "data/input.csv").exists())

    def test_existing_changes_preserved(self):
        target = self.root / "data/input.csv"
        target.write_bytes(b"local changes")
        with self.assertRaisesRegex(ValueError, "Existing input differs"):
            module.restore(self.root)
        self.assertEqual(target.read_bytes(), b"local changes")

    def test_path_escape_rejected(self):
        self.entry["destination"] = "../outside.csv"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "escapes repository"):
            module.restore(self.root)


if __name__ == "__main__":
    unittest.main()

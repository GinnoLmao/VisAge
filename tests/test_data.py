"""Meaningful synthetic checks for data exclusion, grouping and freeze safety."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from visage.data import age_to_label, load_manifest, prepare_dataset, verify_manifest


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.output = self.root / "processed"

    def tearDown(self):
        self.temporary.cleanup()

    def image(self, age, number, pixels=None):
        if pixels is None:
            pixels = np.random.default_rng(number).integers(0, 256, (40, 40, 3), dtype=np.uint8)
        path = self.raw / f"{age}_0_0_{20260101000000000 + number}.jpg.chip.jpg"
        Image.fromarray(pixels).save(path, format="JPEG", quality=95)
        return path

    def population(self, per_class=9):
        for index in range(per_class):
            for label, age in enumerate((12, 35, 70)):
                self.image(age, index * 3 + label)

    def test_boundary_class_mapping(self):
        self.assertEqual([age_to_label(age) for age in (0, 17, 18, 59, 60, 116)], [0, 0, 1, 1, 2, 2])
        for invalid in (-1, 117, 18.0, True):
            with self.assertRaises(ValueError):
                age_to_label(invalid)

    def test_deterministic_group_splits_and_full_verification(self):
        self.population()
        audit = prepare_dataset(self.raw, self.output, phash_distance=0)
        second = self.root / "other"
        prepare_dataset(self.raw, second, phash_distance=0)
        for filename in ("manifest.csv", "excluded.csv", "audit.json", "manifest.freeze.json"):
            self.assertEqual((self.output / filename).read_bytes(), (second / filename).read_bytes())
        self.assertEqual(audit["included_samples"], 27)
        self.assertTrue(all(audit["split_counts"][split] > 0 for split in ("train", "validation", "test")))
        for counts in audit["split_class_counts"].values():
            self.assertTrue(all(count > 0 for count in counts.values()))
        result = verify_manifest(self.output / "manifest.csv", self.raw, verify_files=True)
        self.assertEqual(result["samples"], 27)
        self.assertFalse(result["identity_split_verified"])

    def test_corrupt_bad_names_and_invalid_ages_are_quarantined(self):
        self.population(2)
        (self.raw / "bad_name.jpg").write_bytes(b"not an image")
        (self.raw / "18_0_0_20260101000000999.jpg").write_bytes(b"not an image")
        self.image(117, 900)
        audit = prepare_dataset(self.raw, self.output, phash_distance=0)
        self.assertEqual(audit["included_samples"], 6)
        with (self.output / "excluded.csv").open(newline="", encoding="utf-8") as stream:
            reasons = {row["reason"] for row in csv.DictReader(stream)}
        self.assertEqual(reasons, {"invalid_filename", "unreadable_image", "invalid_age"})

    def test_exact_duplicates_keep_canonical_and_conflicting_groups_are_excluded(self):
        self.population(2)
        shared = np.random.default_rng(1000).integers(0, 256, (40, 40, 3), dtype=np.uint8)
        self.image(10, 1000, shared)
        self.image(11, 1001, shared)
        conflicting = np.random.default_rng(2000).integers(0, 256, (40, 40, 3), dtype=np.uint8)
        self.image(17, 2000, conflicting)
        self.image(18, 2001, conflicting)
        audit = prepare_dataset(self.raw, self.output, phash_distance=0)
        self.assertEqual(audit["included_samples"], 7)
        self.assertEqual(audit["exact_duplicates_removed"], 1)
        self.assertEqual(audit["conflicting_duplicate_groups"], 1)
        rows = load_manifest(self.output / "manifest.csv")
        self.assertNotIn(17, [row["age"] for row in rows])
        self.assertNotIn(18, [row["age"] for row in rows])
        self.assertEqual(len({row["pixel_sha256"] for row in rows}), len(rows))

    def test_similar_images_are_grouped_without_claiming_subject_identity(self):
        self.population(2)
        base = np.random.default_rng(4000).integers(20, 220, (40, 40, 3), dtype=np.uint8)
        self.image(40, 4000, base)
        self.image(41, 4001, base + 8)
        prepare_dataset(self.raw, self.output, phash_distance=4)
        rows = [row for row in load_manifest(self.output / "manifest.csv") if row["age"] in (40, 41)]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["group_id"], rows[1]["group_id"])
        self.assertEqual(rows[0]["split"], rows[1]["split"])

    def test_refreeze_requires_explicit_overwrite(self):
        self.population(2)
        prepare_dataset(self.raw, self.output, phash_distance=0)
        old = (self.output / "manifest.csv").read_bytes()
        prepare_dataset(self.raw, self.output, phash_distance=0)
        self.assertEqual(old, (self.output / "manifest.csv").read_bytes())
        self.image(61, 999)
        with self.assertRaisesRegex(ValueError, "Frozen manifest would change"):
            prepare_dataset(self.raw, self.output, phash_distance=0)
        self.assertEqual(old, (self.output / "manifest.csv").read_bytes())
        audit = prepare_dataset(self.raw, self.output, phash_distance=0, overwrite=True)
        self.assertEqual(audit["included_samples"], 7)

    def test_file_tampering_is_detected(self):
        self.population(2)
        prepare_dataset(self.raw, self.output, phash_distance=0)
        row = load_manifest(self.output / "manifest.csv")[0]
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        Image.fromarray(image).save(self.raw / row["relative_path"], format="JPEG")
        with self.assertRaisesRegex(ValueError, "Frozen .* changed"):
            verify_manifest(self.output / "manifest.csv", self.raw, verify_files=True)

    def test_manifest_path_traversal_is_rejected(self):
        self.population(2)
        prepare_dataset(self.raw, self.output, phash_distance=0)
        path = self.output / "manifest.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            columns, rows = reader.fieldnames, list(reader)
        rows[0]["relative_path"] = "../outside.jpg"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "Unsafe relative path"):
            verify_manifest(path, self.raw)

    def test_audit_tampering_breaks_frozen_checksums(self):
        self.population(2)
        prepare_dataset(self.raw, self.output, phash_distance=0)
        audit = self.output / "audit.json"
        audit.write_text(audit.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify_manifest(self.output / "manifest.csv", self.raw)

    def test_decoded_pixel_limit_is_enforced_before_acceptance(self):
        self.image(30, 1)
        with patch("visage.data.MAX_IMAGE_PIXELS", 100):
            audit = prepare_dataset(self.raw, self.output, phash_distance=0)
        self.assertEqual(audit["included_samples"], 0)
        self.assertEqual(audit["excluded_files"], 1)
        self.assertIn("No usable images remain", " ".join(audit["warnings"]))

    def test_exif_rotation_is_applied_to_audit_dimensions(self):
        pixels = np.random.default_rng(9001).integers(0, 256, (20, 40, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        exif = image.getexif()
        exif[274] = 6  # Rotate 90 degrees clockwise for display.
        image.save(self.raw / "30_0_0_20260101000009001.jpg", exif=exif)
        prepare_dataset(self.raw, self.output, phash_distance=0)
        row = load_manifest(self.output / "manifest.csv")[0]
        self.assertEqual((row["width"], row["height"]), (20, 40))
        self.assertTrue(verify_manifest(self.output / "manifest.csv", self.raw, verify_files=True)["ok"])


if __name__ == "__main__":
    unittest.main()

import copy
from dataclasses import asdict
import importlib.util
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.build_dataset import build_records, SEEDS, DATASET, CSV, ANALYSES, REVIEW
from scripts.validate_dataset import (
    ROOT, ContractValidationError, read_jsonl, validate_dataset_records,
    validate_morphology_records,
)
from turkish_llm_morphology.morphology import Analysis, Morpheme, TurkishMorphAnalyzer
from turkish_llm_morphology.morphology.affix_counter import (
    count_affixes, extract_affixes, to_contract_analysis, v1_scope,
)
from turkish_llm_morphology.morphology.dataset import (
    read_csv, write_csv, review_template, validate_manual_review,
)
from turkish_llm_morphology.morphology.feature_extractor import extract_features


FIXTURE = json.loads((ROOT / "tests/fixtures/zeyrek_0_1_3.json").read_text(encoding="utf-8"))


def fixture_analyses(word):
    return [Analysis(**{**a, "group_boundaries": tuple(a["group_boundaries"]),
                       "morphemes": tuple(Morpheme(**m) for m in a["morphemes"])})
            for a in FIXTURE["words"][word]]


def select(analyses, lemma, pos, tags=None):
    return next(a for a in analyses if a.lemma == lemma and a.pos == pos
                and (tags is None or [m.id for m in a.morphemes] == tags))


class FeatureAndAffixTests(unittest.TestCase):
    def test_negative_past_features(self):
        a = fixture_analyses("gelmediler")[0]
        self.assertEqual(extract_features(a), {
            "tense": "past", "mood": "indicative", "polarity": "negative",
            "person": "third", "number": "plural", "case": None,
            "aspect": None, "possessive": None,
        })

    def test_progressive_aspect_and_root_sound_change(self):
        a = fixture_analyses("gidiyorum")[0]
        self.assertEqual(a.lemma, "git")
        self.assertEqual(a.stem, "gid")
        self.assertEqual(extract_features(a)["aspect"], "progressive")
        self.assertEqual([m["surface"] for m in extract_affixes(a)], ["iyor", "um"])
        self.assertEqual(count_affixes(a), 2)

    def test_possessive_does_not_overwrite_noun_number(self):
        a = fixture_analyses("evlerimizden")[0]
        f = extract_features(a)
        self.assertEqual((f["number"], f["case"], f["person"]), ("plural", "ablative", None))
        self.assertEqual(f["possessive"], {"person": "first", "number": "plural"})
        self.assertEqual(count_affixes(a), 3)
        self.assertNotIn("possessive", to_contract_analysis(a)["morphological_features"])

    def test_overt_counts_and_zero_agreement(self):
        for word, lemma, pos, expected in (
            ("geldim", "gel", "VERB", 2), ("gelmediler", "gel", "VERB", 3),
            ("yazar", "yaz", "VERB", 1), ("kitaba", "kitap", "NOUN", 1),
        ):
            with self.subTest(word=word):
                self.assertEqual(count_affixes(select(fixture_analyses(word), lemma, pos)), expected)

    def test_derivation_is_separate_and_rejected_from_selected_scope(self):
        a = fixture_analyses("evdeki")[0]
        self.assertEqual(count_affixes(a), 1)
        self.assertEqual(extract_affixes(a), [
            {"surface": "de", "label": "Loc", "type": "inflectional"},
            {"surface": "ki", "label": "Rel", "type": "derivational"},
        ])
        with self.assertRaisesRegex(ValueError, "Derivation"):
            v1_scope(a)

    def test_copula_and_compound_tense_are_out_of_scope(self):
        for word in ("evdeydi", "gelmişti"):
            for a in fixture_analyses(word):
                with self.subTest(word=word), self.assertRaises(ValueError):
                    v1_scope(a)

    def test_unsupported_mood_does_not_fabricate_a_tense(self):
        f = extract_features(fixture_analyses("gelse")[0])
        self.assertIsNone(f["mood"])
        self.assertIsNone(f["tense"])
        with self.assertRaises(ValueError):
            v1_scope(fixture_analyses("gelse")[0])

    def test_necessitative_has_no_invented_tense(self):
        f = extract_features(fixture_analyses("gelmeliyim")[0])
        self.assertEqual(f["mood"], "necessitative")
        self.assertIsNone(f["tense"])


@unittest.skipUnless(importlib.util.find_spec("zeyrek"), "install .[morphology] for real Zeyrek tests")
class RealMorphologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger("zeyrek.rulebasedanalyzer").setLevel(logging.ERROR)
        cls.analyzer = TurkishMorphAnalyzer()
        cls.records = read_jsonl(DATASET)

    def test_real_outputs_match_captured_objects(self):
        for word in FIXTURE["words"]:
            with self.subTest(word=word):
                actual = json.loads(json.dumps([asdict(a) for a in self.analyzer.analyze(word)]))
                self.assertEqual(actual, FIXTURE["words"][word])

    def test_ambiguous_word_retains_all_four_analyses(self):
        analyses = self.analyzer.analyze("yazar")
        self.assertEqual(len(analyses), 4)
        self.assertEqual({a.pos for a in analyses}, {"NOUN", "ADJ", "VERB"})
        self.assertTrue(any(m.derivational for a in analyses for m in a.morphemes))

    def test_word_order_does_not_poison_stem_attributes(self):
        before = self.analyzer.analyze("geliyorum")
        self.assertTrue(before)
        self.analyzer.analyze("geldik")
        self.assertEqual(self.analyzer.analyze("geliyorum"), before)
        self.assertEqual(TurkishMorphAnalyzer().analyze("geliyorum"), before)

    def test_complete_pilot_is_independent_of_analysis_order(self):
        forward = {r["target"]: self.analyzer.analyze(r["target"]) for r in self.records}
        for r in reversed(self.records):
            self.assertEqual(self.analyzer.analyze(r["target"]), forward[r["target"]])

    def test_empty_invalid_and_non_nfc_input(self):
        for word in (None, "", " ", "gel dim", "gel1", "gel!", " gel", "go\u0308r"):
            with self.subTest(word=word), self.assertRaises(ValueError):
                self.analyzer.analyze(word)

    def test_unknown_word_is_not_fabricated(self):
        self.assertEqual(self.analyzer.analyze("zzzzq"), [])

    def test_saved_pilot_matches_seeds_and_real_analyses(self):
        records, evidence = build_records(read_jsonl(SEEDS), self.analyzer)
        self.assertEqual(records, self.records)
        self.assertEqual(json.loads(json.dumps(evidence)), read_jsonl(ANALYSES))
        validate_morphology_records(records, self.analyzer)
        self.assertEqual(len(records), 40)
        self.assertEqual(len({r["lemma"] for r in records}), 10)
        self.assertEqual(read_csv(CSV), records)

    def test_schema_errors_are_rejected(self):
        mutations = [
            lambda r: r.pop("target"), lambda r: r.update(pos="INVALID"),
            lambda r: r["scope"].update(has_derivation=True),
            lambda r: r.update(target=""), lambda r: r.update(affix_count=-1),
            lambda r: r.update(selected_analysis_index=-1),
            lambda r: r.update(selected_analysis_index=999),
            lambda r: r.update(lemma="wrong"), lambda r: r.update(affix_count=99),
        ]
        for mutate in mutations:
            r = copy.deepcopy(self.records[0])
            mutate(r)
            with self.subTest(record=r), self.assertRaises(ContractValidationError):
                validate_morphology_records([r], self.analyzer)

    def test_duplicate_ids_and_duplicate_examples_are_rejected(self):
        first = self.records[0]
        second = copy.deepcopy(first)
        with self.assertRaisesRegex(ContractValidationError, "duplicate sample_id"):
            validate_morphology_records([first, second], self.analyzer)
        second["sample_id"] = "new-duplicate"
        with self.assertRaisesRegex(ContractValidationError, "duplicate prefix/target"):
            validate_morphology_records([first, second], self.analyzer)

    def test_empty_dataset_or_bad_prefix_is_rejected(self):
        with self.assertRaises(ContractValidationError):
            validate_morphology_records([], self.analyzer)
        for prefix in ("", " ", "Dün", "Dün  ", "Dün\t "):
            r = {**self.records[0], "prefix": prefix}
            with self.subTest(prefix=prefix), self.assertRaises(ContractValidationError):
                validate_morphology_records([r], self.analyzer)

    def test_coherently_wrong_lemma_is_caught_against_analyzer(self):
        r = copy.deepcopy(self.records[0])
        r["lemma"] = "wrong"
        r["morphological_analyses"][r["selected_analysis_index"]]["lemma"] = "wrong"
        validate_dataset_records([r])
        with self.assertRaisesRegex(ContractValidationError, "differ from Zeyrek"):
            validate_morphology_records([r], self.analyzer)

    def test_lost_alternatives_are_caught_even_when_shared_schema_passes(self):
        r = copy.deepcopy(next(r for r in self.records if r["target"] == "yazar"))
        r["morphological_analyses"] = [r["morphological_analyses"][r["selected_analysis_index"]]]
        r["selected_analysis_index"] = 0
        validate_dataset_records([r])
        with self.assertRaisesRegex(ContractValidationError, "alternatives/fields lost"):
            validate_morphology_records([r], self.analyzer)

    def test_false_affix_surface_is_caught_against_analyzer(self):
        r = copy.deepcopy(self.records[0])
        r["morphological_analyses"][r["selected_analysis_index"]]["affixes"][0]["surface"] = "fake"
        validate_dataset_records([r])
        with self.assertRaises(ContractValidationError):
            validate_morphology_records([r], self.analyzer)

    def test_tokenizer_alignment_error_is_propagated(self):
        with patch("turkish_llm_morphology.token_boundary.check_alignment", return_value={
                "status": "alignment_error", "error": {"message": "boundary crossing"}}):
            with self.assertRaisesRegex(ContractValidationError, "boundary crossing"):
                validate_morphology_records(self.records[:1], self.analyzer, tokenizer=object())

    def test_coherently_wrong_affix_count_is_caught_against_analyzer(self):
        r = copy.deepcopy(self.records[0])
        r["affix_count"] += 1
        r["morphological_analyses"][r["selected_analysis_index"]]["affixes"].append(
            {"surface": "fake", "label": "Fake", "type": "inflectional"})
        validate_dataset_records([r])
        with self.assertRaisesRegex(ContractValidationError, "differ from Zeyrek"):
            validate_morphology_records([r], self.analyzer)


class DatasetIOAndReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = read_jsonl(DATASET)

    def test_csv_json_nested_fields_and_prefix_whitespace_round_trip(self):
        records = copy.deepcopy(self.records[:1])
        records[0]["prefix"] = 'Alıntı: "İçeri, gel" dedi; şimdi '
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "roundtrip.csv"
            write_csv(records, path)
            self.assertEqual(read_csv(path), records)

    def test_bad_csv_columns_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("sample_id,target\na,b\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "header"):
                read_csv(path)

    def test_review_manifest_is_bound_to_the_committed_records(self):
        review = json.loads(REVIEW.read_text(encoding="utf-8"))
        expected = review_template(self.records)
        self.assertEqual([(r["sample_id"], r["sample_sha256"]) for r in review["samples"]],
                         [(r["sample_id"], r["sample_sha256"]) for r in expected["samples"]])

    def test_review_template_never_marks_samples_approved(self):
        template = review_template(self.records)
        self.assertTrue(all(r["status"] == "pending" and r["reviewer"] is None
                            for r in template["samples"]))
        with self.assertRaisesRegex(ValueError, "Human review pending"):
            validate_manual_review(self.records, template)

    def test_review_checks_count_identity_dates_and_content_hashes(self):
        # Synthetic attestations exercise validation; they are never saved or used for the pilot.
        review = review_template(self.records)
        for entry in review["samples"]:
            entry.update(status="approved", reviewer="TEST FIXTURE ONLY", reviewed_at="2026-11-24")
        validate_manual_review(self.records, review)
        for field, value in (("sample_sha256", "stale"), ("reviewer", ""), ("reviewed_at", "invalid")):
            invalid = copy.deepcopy(review)
            invalid["samples"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_manual_review(self.records, invalid)
        with self.assertRaises(ValueError):
            validate_manual_review(self.records[:29], review_template(self.records[:29]))
        invalid = copy.deepcopy(review)
        invalid["samples"].append(invalid["samples"][0])
        with self.assertRaises(ValueError):
            validate_manual_review(self.records, invalid)


if __name__ == "__main__":
    unittest.main()

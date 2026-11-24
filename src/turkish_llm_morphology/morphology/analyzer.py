"""Adapter for Zeyrek 0.1.3's structured SingleAnalysis objects."""

from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version
from threading import RLock
import unicodedata


POS = {
    "Noun": "NOUN", "Adj": "ADJ", "Adv": "ADV", "Verb": "VERB",
    "Pron": "PRON", "Num": "NUM", "Det": "DET", "Conj": "CCONJ",
    "Interj": "INTJ", "Postp": "ADP", "Ques": "PART", "Punc": "PUNCT",
    "Dup": "X", "Unk": "X",
}
_BACKEND_LOCK = RLock()


@lru_cache(maxsize=1)
def _load_backend():
    if version("zeyrek") != "0.1.3":
        raise RuntimeError("This SingleAnalysis adapter requires zeyrek==0.1.3")
    import zeyrek

    # Zeyrek's graph uses module-level states; rebuilding it appends duplicate edges.
    return zeyrek.MorphAnalyzer()


@dataclass(frozen=True)
class Morpheme:
    id: str
    surface: str
    derivational: bool


@dataclass(frozen=True)
class Analysis:
    word: str
    dictionary_id: str
    dictionary_lemma: str
    dictionary_root: str
    dictionary_pos: str
    pos: str
    stem: str
    ending: str
    morphemes: tuple[Morpheme, ...]
    group_boundaries: tuple[int, ...]

    @property
    def lemma(self) -> str:
        # The v1 pilot uses verb roots (gel), while Zeyrek lemmas are infinitives (gelmek).
        return self.dictionary_root if self.dictionary_pos == "VERB" else self.dictionary_lemma


class TurkishMorphAnalyzer:
    def __init__(self):
        with _BACKEND_LOCK:
            self._analyzer = _load_backend()

    def analyze(self, word: str) -> list[Analysis]:
        """Keep every analysis, including out-of-scope alternatives; unknown words return []."""
        if not isinstance(word, str) or not word or not word.isalpha():
            raise ValueError("word must be one non-empty alphabetic word without whitespace")
        if word != unicodedata.normalize("NFC", word):
            raise ValueError("word must use Unicode NFC")
        from zeyrek.attributes import calculate_phonetic_attributes
        from zeyrek.morphology import _normalize

        # Public Parse drops suffix surfaces. _parse returns real (Morpheme, surface)
        # pairs and avoids sentence tokenization / an unrelated NLTK punkt download.
        with _BACKEND_LOCK:
            # Upstream mutates cached sets and stem attrs during search. Isolate each
            # word so geldik cannot remove later geliyorum analyses (PastPart ~k).
            calculate_phonetic_attributes.cache_clear()
            transitions = self._analyzer.analyzer.stem_transitions.prefix_matches(_normalize(word))
            originals = [(transition, transition.attrs) for transition in transitions]
            for transition, attrs in originals:
                transition.attrs = attrs.copy()
            try:
                raw_analyses = self._analyzer._parse(word)
            finally:
                for transition, attrs in originals:
                    transition.attrs = attrs
        analyses = []
        for raw in raw_analyses:
            pos = POS[raw.pos.value]
            if pos == "NOUN" and raw.dict_item.secondary_pos.value == "Prop":
                pos = "PROPN"
            analyses.append(Analysis(
                word=word, dictionary_id=raw.dict_item.id_,
                dictionary_lemma=raw.dict_item.lemma, dictionary_root=raw.dict_item.root,
                dictionary_pos=POS[raw.dict_item.primary_pos.value], pos=pos,
                stem=raw.stem, ending=raw.ending,
                morphemes=tuple(Morpheme(m.id_, surface, m.derivational)
                                for m, surface in raw.morphemes),
                group_boundaries=tuple(raw.group_boundaries),
            ))
        # Lexicon iteration can depend on Python hash order. Never deduplicate alternatives.
        return sorted(analyses, key=lambda a: (
            a.dictionary_id, a.pos,
            tuple((m.id, m.surface, m.derivational) for m in a.morphemes),
        ))

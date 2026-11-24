"""Count overt inflectional segments; keep derivation separate."""

from .feature_extractor import TENSE, contract_features


def extract_affixes(analysis) -> list[dict]:
    return [
        {"surface": m.surface, "label": m.id,
         "type": "derivational" if m.derivational else "inflectional"}
        for m in analysis.morphemes[1:] if m.surface
    ]


def count_affixes(analysis) -> int:
    return sum(affix["type"] == "inflectional" for affix in extract_affixes(analysis))


def to_contract_analysis(analysis) -> dict:
    return {"lemma": analysis.lemma, "pos": analysis.pos,
            "morphological_features": contract_features(analysis),
            "affixes": extract_affixes(analysis)}


def v1_scope(analysis) -> dict:
    """Reject unsupported selected analyses; alternatives remain untouched."""
    tags = [m.id for m in analysis.morphemes]
    if any(m.derivational for m in analysis.morphemes[1:]):
        raise ValueError("Derivation/zero conversion is outside the v1 pilot scope")
    if "Cop" in tags or tags.count("Verb") > 1 or sum(t in TENSE for t in tags) > 1:
        raise ValueError("Copula/compound tense is outside the v1 pilot scope")
    if "Ques" in tags:
        raise ValueError("Clitics are outside the v1 pilot scope")
    features = contract_features(analysis)
    if count_affixes(analysis) == 0:
        raise ValueError("An overt inflectional affix is required by the v1 pilot")
    if analysis.pos == "VERB" and all(features[k] is not None for k in (
            "tense", "mood", "polarity", "person", "number")):
        experiment_type = "simple_verb_inflection"
    elif analysis.pos in {"NOUN", "PROPN"} and features["case"] not in {None, "nominative"}:
        experiment_type = "noun_case"
    else:
        raise ValueError("Analysis is not a supported finite verb or case-marked noun")
    return {"experiment_type": experiment_type, "has_derivation": False,
            "has_copula": False, "has_clitic": False, "is_multiword_target": False}

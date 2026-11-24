"""Features from morpheme IDs, scoped to the final inflectional group."""

CONTRACT_FEATURES = ("tense", "mood", "polarity", "person", "number", "case")
TENSE = {
    "Past": "past", "Pres": "present", "Prog1": "progressive",
    "Prog2": "progressive", "Aor": "aorist", "Fut": "future", "Narr": "evidential",
}
MOOD = {"Imp": "imperative", "Cond": "conditional", "Neces": "necessitative",
        "Opt": "optative"}
CASE = {"Nom": "nominative", "Acc": "accusative", "Dat": "dative",
        "Loc": "locative", "Abl": "ablative", "Gen": "genitive",
        "Ins": "instrumental", "Equ": "equative"}
AGREEMENT = {
    "1sg": ("first", "singular"), "2sg": ("second", "singular"),
    "3sg": ("third", "singular"), "1pl": ("first", "plural"),
    "2pl": ("second", "plural"), "3pl": ("third", "plural"),
}


def extract_features(analysis) -> dict:
    features = dict.fromkeys((*CONTRACT_FEATURES, "aspect", "possessive"))
    tags = [m.id for m in analysis.morphemes[analysis.group_boundaries[-1]:]]
    for tag in tags:
        if tag.startswith("A") and tag[1:] in AGREEMENT:
            person, number = AGREEMENT[tag[1:]]
            features["number"] = number
            if analysis.pos in {"VERB", "PRON"}:
                features["person"] = person
        if tag.startswith("P") and tag[1:] in AGREEMENT:
            person, number = AGREEMENT[tag[1:]]
            features["possessive"] = {"person": person, "number": number}
        if analysis.pos == "VERB":
            if tag in TENSE:
                features["tense"] = TENSE[tag]
            if tag in MOOD:
                features["mood"] = MOOD[tag]
            if tag in {"Prog1", "Prog2"}:
                features["aspect"] = "progressive"
        elif tag in CASE:
            features["case"] = CASE[tag]
    if analysis.pos == "VERB":
        features["polarity"] = "negative" if "Neg" in tags else "positive"
        if features["mood"] is None and features["tense"] is not None:
            features["mood"] = "indicative"
    elif analysis.pos in {"NOUN", "PROPN", "PRON"} and features["case"] is None:
        # Zeyrek 0.1.3's parse_analysis explicitly omits the Nom and Pnon transitions.
        features["case"] = "nominative"
    return features


def contract_features(analysis) -> dict:
    features = extract_features(analysis)
    return {key: features[key] for key in CONTRACT_FEATURES}

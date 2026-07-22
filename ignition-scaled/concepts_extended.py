"""40 additional concepts for the scaled ignition run: same word families as the
original 40 (countries + concrete nouns spanning animals, household objects,
instruments, vehicles, food, nature). Names only — arm 1 (prompt ladders) runs the
original 40 unchanged; the extension feeds arm 2 (injection) and the competition
matrix, which need only a single-token name and a kind.

All 40 verified single-token with leading space on the Qwen3-8B tokenizer
(2026-07-21); SPARES are fallbacks for other tokenizers. The runner re-filters at
startup and records survivors/drops in the manifest.
"""

EXTRA_COUNTRIES = [
    "Turkey", "Greece", "Sweden", "Norway", "Poland", "Portugal",
    "Australia", "Argentina", "Iran", "Kenya", "Thailand", "Vietnam",
]

EXTRA_NOUNS = [
    "wolf", "bear", "tiger", "rabbit", "monkey", "whale", "frog", "owl",
    "lamp", "table", "chair", "bottle",
    "violin", "drum", "flute",
    "car", "truck", "rocket", "ship",
    "rice", "soup", "honey", "butter", "lemon",
    "river", "mountain", "cloud", "snow",
]

SPARE_COUNTRIES = ["Chile", "Peru", "Ireland", "Austria"]
SPARE_NOUNS = ["tree", "stone", "glass", "bridge", "tower", "island", "forest", "storm"]


def extra_concepts():
    """name -> kind, in deterministic order."""
    out = {}
    for n in EXTRA_COUNTRIES:
        out[n] = "country"
    for n in EXTRA_NOUNS:
        out[n] = "noun"
    return out


def resident_carrier(name, kind):
    """Competition-matrix resident carrier: evokes Y by naming it."""
    if kind == "country":
        return f"The story was about {name}."
    return f"The story was about the {name}."

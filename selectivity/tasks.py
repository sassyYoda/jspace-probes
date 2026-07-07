"""Task battery for the selectivity dose-response check.

Two scoring types: "gen" (teacher-forced greedy match against answer variants)
and "logit" (compare option-token logits at the last position; no generation).
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
import os
PROBE_SWAP = Path(os.environ.get("JLENS_REPO", ROOT / "../vendor/jacobian-lens")) / "data/experiments/probe-swap.json"

RHYME = [
    ("cat", "a long-tailed rodent", "rat"),
    ("dog", "a small amphibian that hops", "frog"),
    ("house", "a small rodent chased by cats", "mouse"),
    ("king", "a circular band worn on a finger", "ring"),
    ("day", "dried grass used to feed horses", "hay"),
    ("bear", "a seat with four legs and a back", "chair"),
    ("moon", "an eating utensil used for soup", "spoon"),
    ("star", "a vehicle with four wheels", "car"),
    ("rain", "a locomotive pulling carriages", "train"),
    ("bell", "the hard outer covering of an egg", "shell"),
    ("cake", "a long limbless reptile", "snake"),
    ("tree", "a stinging insect that makes honey", "bee"),
    ("goat", "a small vessel for traveling on water", "boat"),
    ("wall", "a round object used in games", "ball"),
    ("sun", "a sweet baked bread roll", "bun"),
    ("hat", "a flying mammal active at night", "bat"),
    ("sock", "a large solid piece of stone", "rock"),
    ("gold", "the opposite of hot", "cold"),
    ("red", "a piece of furniture for sleeping", "bed"),
    ("book", "a person who prepares food", "cook"),
    ("mouse", "a building where people live", "house"),
    ("night", "a flying toy on a string", "kite"),
    ("door", "the surface you walk on indoors", "floor"),
    ("green", "a flat surface that displays images", "screen"),
    ("light", "the opposite of day", "night"),
    ("snow", "a weapon that shoots arrows", "bow"),
    ("mail", "a slow-moving creature with a shell", "snail"),
    ("pen", "a female chicken", "hen"),
    ("sea", "a hot drink made from leaves", "tea"),
    ("lamp", "a small piece of paper stuck on a letter", "stamp"),
    ("coat", "a farm animal that eats almost anything", "goat"),
    ("fish", "a shallow container for serving food", "dish"),
    ("dock", "a device that tells the time", "clock"),
    ("spoon", "midday", "noon"),
    ("cash", "refuse or garbage", "trash"),
    ("dream", "the thick fatty part of milk", "cream"),
    ("plane", "water falling from the sky", "rain"),
    ("france", "rhythmic movement to music", "dance"),
    ("kite", "the source of illumination", "light"),
    ("fun", "to move quickly on foot", "run"),
]

CAPITALS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"),
    ("Spain", "Madrid"), ("Germany", "Berlin"), ("Russia", "Moscow"),
    ("England", "London"), ("Egypt", "Cairo"), ("Greece", "Athens"),
    ("Portugal", "Lisbon"), ("China", "Beijing"), ("Canada", "Ottawa"),
    ("Australia", "Canberra"), ("Poland", "Warsaw"), ("Austria", "Vienna"),
    ("Ireland", "Dublin"), ("Norway", "Oslo"), ("Sweden", "Stockholm"),
    ("Kenya", "Nairobi"), ("Turkey", "Ankara"),
]

FACTS = [
    ("Fact: The largest planet in the solar system is", ["Jupiter"]),
    ("Fact: The planet closest to the Sun is", ["Mercury"]),
    ("Fact: The tallest mountain on Earth is", ["Mount Everest", "Everest"]),
    ("Fact: The largest ocean on Earth is the", ["Pacific"]),
    ("Fact: The longest river in South America is the", ["Amazon"]),
    ("Fact: The author of Hamlet is", ["William Shakespeare", "Shakespeare"]),
    ("Fact: The scientist who developed the theory of relativity is",
     ["Albert Einstein", "Einstein"]),
    ("Fact: The first president of the United States was",
     ["George Washington"]),
    ("Fact: The largest desert in Africa is the", ["Sahara"]),
    ("Fact: The continent where Brazil is located is", ["South America"]),
    ("Fact: The language spoken in Mexico is", ["Spanish"]),
    ("Fact: The currency used in the United States is the", ["dollar", "US"]),
    ("Fact: The animal known as the king of the jungle is the", ["lion"]),
    ("Fact: The fastest land animal is the", ["cheetah"]),
    ("Fact: The largest mammal on Earth is the", ["blue whale", "whale"]),
    ("Fact: The painter of the Mona Lisa is",
     ["Leonardo da Vinci", "Leonardo"]),
    ("Fact: The number of days in a week is", ["seven", "7"]),
    ("Fact: The color of the sky on a clear day is", ["blue"]),
    ("Fact: The gas that humans need to breathe is", ["oxygen"]),
    ("Fact: The frozen form of water is called", ["ice"]),
]

ARITH_TEMPLATES = [
    "If you add {a} and {b} and then multiply the result by {c}, you get",
    "Take {a}, add {b}, and multiply the sum by {c}. The result is",
]


def arithmetic_items(n, seed):
    rng = random.Random(seed)
    seen, items = set(), []
    while len(items) < n:
        a, b, c = rng.randint(3, 9), rng.randint(3, 9), rng.randint(2, 5)
        if (a, b, c) in seen:
            continue
        seen.add((a, b, c))
        t = ARITH_TEMPLATES[len(items) % len(ARITH_TEMPLATES)]
        items.append({
            "task": "arithmetic", "id": f"arith-{len(items)}",
            "prompt": t.format(a=a, b=b, c=c), "type": "gen",
            "answers": [str((a + b) * c)],
        })
    return items


def build_battery(cfg, tok):
    n = cfg["n_items"]
    items = []

    for it in json.load(open(PROBE_SWAP))["items"]:
        items.append({
            "task": "multihop", "id": it["name"],
            "prompt": it["prompt"].rstrip(),  # trailing-space gotcha
            "type": "gen", "answers": [it["answer"]],
        })

    items += arithmetic_items(n, cfg["arith_seed"])

    for i, (cue, defn, target) in enumerate(RHYME[:n]):
        items.append({
            "task": "rhyme", "id": f"rhyme-{i}",
            "prompt": f"Q: What word rhymes with {cue} and means {defn}?\nA: The word is",
            "type": "gen",
            # model wraps answers in quotes/bold; accept those greedy paths
            "answers": [target, target.capitalize(), f'"{target}', f"**{target}",
                        f'"{target.capitalize()}', f"**{target.capitalize()}"],
        })

    from datasets import load_dataset

    mmlu = load_dataset("cais/mmlu", "all", split="validation")
    rng = random.Random(cfg["mmlu_seed"])
    idx = [i for i in rng.sample(range(len(mmlu)), 400)
           if len(mmlu[i]["question"]) + sum(len(c) for c in mmlu[i]["choices"]) < 500][:n]
    letters = ["A", "B", "C", "D"]
    opt_ids = [tok.encode(" " + L, add_special_tokens=False)[0] for L in letters]
    for i in idx:
        ex = mmlu[i]
        lines = [f"Question: {ex['question'].strip()}"]
        lines += [f"{L}) {c}" for L, c in zip(letters, ex["choices"])]
        lines.append("Answer:")
        items.append({
            "task": "mcqa", "id": f"mmlu-{i}", "prompt": "\n".join(lines),
            "type": "logit", "options": opt_ids, "label": ex["answer"],
        })

    sst = load_dataset("stanfordnlp/sst2", split="validation")
    rng = random.Random(cfg["sst2_seed"])
    idx = [i for i in rng.sample(range(len(sst)), 200)
           if 20 < len(sst[i]["sentence"]) < 200][:n]
    opt_ids = [tok.encode(" negative", add_special_tokens=False)[0],
               tok.encode(" positive", add_special_tokens=False)[0]]
    for i in idx:
        ex = sst[i]
        items.append({
            "task": "sentiment", "id": f"sst2-{i}",
            "prompt": f"Review: {ex['sentence'].strip()}\nSentiment (positive/negative):",
            "type": "logit", "options": opt_ids, "label": ex["label"],
        })

    single = [{"task": "singlehop", "id": f"cap-{i}",
               "prompt": f"Fact: The capital of {c} is", "type": "gen",
               "answers": [a]} for i, (c, a) in enumerate(CAPITALS)]
    single += [{"task": "singlehop", "id": f"fact-{i}", "prompt": p,
                "type": "gen", "answers": ans} for i, (p, ans) in enumerate(FACTS)]
    items += single[:n]

    return items

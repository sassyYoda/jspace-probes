"""Item construction for the D-redesign fixed-option test.

Three task types (singlehop, multihop, rhyme) x 60 underlying questions each,
every question emitted in TWO scoring formats:

- gen:   open completion, teacher-forced exact match (signal-check style).
- fixed: 4-option A-D multiple choice over the correct answer + 3 same-family
         distractors; scored by letter-token logit comparison. The answer's
         content word appears ONLY in the options list (part of the prompt), so
         the prompt-literal exclusion rule keeps its direction out of the
         ablatable active set.

Everything seeded. Distractors are hand-authored from the same word family as
the answer (other capitals, other rhymes of the same cue, other planets, ...).
"""

import json
import os
import random
import zlib
from pathlib import Path


def _rng(*key):
    """Process-independent seeded rng (string hash() is salted; crc32 is not)."""
    return random.Random(zlib.crc32("|".join(map(str, key)).encode()))

ROOT = Path(__file__).resolve().parent
PROBE_SWAP = Path(os.environ.get("JLENS_REPO", ROOT / "../vendor/jacobian-lens")) \
    / "data/experiments/probe-swap.json"

LETTERS = ["A", "B", "C", "D"]

# ---------------------------------------------------------------- rhyme (60)
# (cue, definition, answer, [3 distractors that rhyme with cue, wrong meaning])
RHYME = [
    ("cat", "a long-tailed rodent", "rat", ["bat", "hat", "mat"]),
    ("dog", "a small amphibian that hops", "frog", ["log", "fog", "bog"]),
    ("house", "a small rodent chased by cats", "mouse", ["blouse", "spouse", "louse"]),
    ("king", "a circular band worn on a finger", "ring", ["wing", "string", "thing"]),
    ("day", "dried grass used to feed horses", "hay", ["bay", "clay", "tray"]),
    ("bear", "a seat with four legs and a back", "chair", ["pear", "stair", "hair"]),
    ("moon", "an eating utensil used for soup", "spoon", ["balloon", "raccoon", "noon"]),
    ("star", "a vehicle with four wheels", "car", ["bar", "jar", "tar"]),
    ("rain", "a locomotive pulling carriages", "train", ["brain", "chain", "drain"]),
    ("bell", "the hard outer covering of an egg", "shell", ["well", "spell", "cell"]),
    ("cake", "a long limbless reptile", "snake", ["lake", "rake", "flake"]),
    ("tree", "a stinging insect that makes honey", "bee", ["key", "knee", "flea"]),
    ("goat", "a small vessel for traveling on water", "boat", ["coat", "moat", "note"]),
    ("wall", "a round object used in games", "ball", ["hall", "fall", "mall"]),
    ("sun", "a sweet baked bread roll", "bun", ["fun", "gun", "run"]),
    ("hat", "a flying mammal active at night", "bat", ["rat", "mat", "flat"]),
    ("sock", "a large solid piece of stone", "rock", ["block", "clock", "lock"]),
    ("gold", "the opposite of hot", "cold", ["bold", "mold", "fold"]),
    ("red", "a piece of furniture for sleeping", "bed", ["shed", "sled", "thread"]),
    ("book", "a person who prepares food", "cook", ["hook", "brook", "nook"]),
    ("mouse", "a building where people live", "house", ["blouse", "spouse", "louse"]),
    ("night", "a flying toy on a string", "kite", ["light", "bite", "site"]),
    ("door", "the surface you walk on indoors", "floor", ["shore", "core", "store"]),
    ("green", "a flat surface that displays images", "screen", ["queen", "bean", "machine"]),
    ("light", "the opposite of day", "night", ["kite", "bite", "height"]),
    ("snow", "a weapon that shoots arrows", "bow", ["crow", "glow", "dough"]),
    ("mail", "a slow-moving creature with a shell", "snail", ["whale", "tail", "pail"]),
    ("pen", "a female chicken", "hen", ["den", "ten", "wren"]),
    ("sea", "a hot drink made from leaves", "tea", ["bee", "key", "knee"]),
    ("lamp", "a small piece of paper stuck on a letter", "stamp", ["camp", "ramp", "clamp"]),
    ("coat", "a farm animal that eats almost anything", "goat", ["boat", "moat", "note"]),
    ("fish", "a shallow container for serving food", "dish", ["wish", "swish", "squish"]),
    ("dock", "a device that tells the time", "clock", ["rock", "block", "lock"]),
    ("spoon", "midday", "noon", ["balloon", "raccoon", "cartoon"]),
    ("cash", "refuse or garbage", "trash", ["flash", "splash", "crash"]),
    ("dream", "the thick fatty part of milk", "cream", ["steam", "beam", "stream"]),
    ("plane", "water falling from the sky", "rain", ["chain", "brain", "cane"]),
    ("france", "rhythmic movement to music", "dance", ["chance", "glance", "trance"]),
    ("kite", "the source of illumination", "light", ["bite", "site", "height"]),
    ("fun", "to move quickly on foot", "run", ["bun", "gun", "ton"]),
    ("town", "a golden headpiece worn by a monarch", "crown", ["gown", "clown", "frown"]),
    ("car", "a great distance away", "far", ["bar", "jar", "tar"]),
    ("mice", "frozen water", "ice", ["dice", "rice", "spice"]),
    ("bake", "a large body of fresh water", "lake", ["rake", "cake", "flake"]),
    ("tall", "the season when the leaves drop", "fall", ["ball", "hall", "wall"]),
    ("deep", "a farm animal with wool", "sheep", ["jeep", "heap", "creep"]),
    ("money", "a sweet substance made by bees", "honey", ["bunny", "sunny", "funny"]),
    ("chain", "the organ inside your head used for thinking", "brain", ["rain", "train", "drain"]),
    ("cry", "the region above the earth where clouds float", "sky", ["pie", "fly", "tie"]),
    ("bark", "a large gray fish with sharp teeth", "shark", ["park", "dark", "spark"]),
    ("wing", "the season that follows winter", "spring", ["ring", "king", "thing"]),
    ("mine", "a tall evergreen tree with needles", "pine", ["vine", "line", "spine"]),
    ("bone", "a small piece of rock", "stone", ["throne", "cone", "phone"]),
    ("jail", "a very strong wind", "gale", ["whale", "tail", "pail"]),
    ("toast", "a spirit said to haunt houses", "ghost", ["coast", "roast", "host"]),
    ("sword", "a flat piece of wood", "board", ["cord", "lord", "hoard"]),
    ("spice", "small cubes with dots used in games", "dice", ["rice", "mice", "ice"]),
    ("street", "the warmth given off by a fire", "heat", ["meat", "seat", "wheat"]),
    ("glass", "the green plants covering a lawn", "grass", ["brass", "class", "pass"]),
    ("blue", "the footwear worn on a foot", "shoe", ["glue", "crew", "stew"]),
]

# ------------------------------------------------------------ singlehop (60)
CAPITALS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"),
    ("Spain", "Madrid"), ("Germany", "Berlin"), ("Russia", "Moscow"),
    ("England", "London"), ("Egypt", "Cairo"), ("Greece", "Athens"),
    ("Portugal", "Lisbon"), ("China", "Beijing"), ("Canada", "Ottawa"),
    ("Australia", "Canberra"), ("Poland", "Warsaw"), ("Austria", "Vienna"),
    ("Ireland", "Dublin"), ("Norway", "Oslo"), ("Sweden", "Stockholm"),
    ("Kenya", "Nairobi"), ("Turkey", "Ankara"),
    ("Netherlands", "Amsterdam"), ("Thailand", "Bangkok"),
    ("Hungary", "Budapest"), ("Finland", "Helsinki"), ("Cuba", "Havana"),
    ("Peru", "Lima"), ("Iceland", "Reykjavik"), ("Argentina", "Buenos Aires"),
    ("India", "New Delhi"), ("Switzerland", "Bern"),
]

# (prompt, [gen answer variants], canonical option, [3 same-family distractors])
FACTS = [
    ("Fact: The largest planet in the solar system is", ["Jupiter"],
     "Jupiter", ["Saturn", "Neptune", "Mars"]),
    ("Fact: The planet closest to the Sun is", ["Mercury"],
     "Mercury", ["Venus", "Mars", "Jupiter"]),
    ("Fact: The tallest mountain on Earth is", ["Mount Everest", "Everest"],
     "Everest", ["Kilimanjaro", "Fuji", "Denali"]),
    ("Fact: The largest ocean on Earth is the", ["Pacific"],
     "Pacific", ["Atlantic", "Indian", "Arctic"]),
    ("Fact: The longest river in South America is the", ["Amazon"],
     "Amazon", ["Nile", "Mississippi", "Congo"]),
    ("Fact: The author of Hamlet is", ["William Shakespeare", "Shakespeare"],
     "Shakespeare", ["Dickens", "Chaucer", "Milton"]),
    ("Fact: The scientist who developed the theory of relativity is",
     ["Albert Einstein", "Einstein"], "Einstein", ["Newton", "Bohr", "Darwin"]),
    ("Fact: The first president of the United States was", ["George Washington"],
     "George Washington", ["John Adams", "Thomas Jefferson", "Abraham Lincoln"]),
    ("Fact: The largest desert in Africa is the", ["Sahara"],
     "Sahara", ["Kalahari", "Gobi", "Namib"]),
    ("Fact: The continent where Brazil is located is", ["South America"],
     "South America", ["North America", "Africa", "Europe"]),
    ("Fact: The language spoken in Mexico is", ["Spanish"],
     "Spanish", ["Portuguese", "French", "Italian"]),
    ("Fact: The currency used in the United States is the", ["dollar", "US"],
     "dollar", ["peso", "euro", "pound"]),
    ("Fact: The animal known as the king of the jungle is the", ["lion"],
     "lion", ["tiger", "elephant", "bear"]),
    ("Fact: The fastest land animal is the", ["cheetah"],
     "cheetah", ["leopard", "gazelle", "horse"]),
    ("Fact: The largest mammal on Earth is the", ["blue whale", "whale"],
     "blue whale", ["elephant", "giraffe", "hippopotamus"]),
    ("Fact: The painter of the Mona Lisa is", ["Leonardo da Vinci", "Leonardo"],
     "Leonardo da Vinci", ["Michelangelo", "Raphael", "Donatello"]),
    ("Fact: The number of days in a week is", ["seven", "7"],
     "seven", ["five", "six", "eight"]),
    ("Fact: The color of the sky on a clear day is", ["blue"],
     "blue", ["green", "red", "yellow"]),
    ("Fact: The gas that humans need to breathe is", ["oxygen"],
     "oxygen", ["nitrogen", "hydrogen", "carbon"]),
    ("Fact: The frozen form of water is called", ["ice"],
     "ice", ["steam", "snow", "frost"]),
    ("Fact: The planet known as the red planet is", ["Mars"],
     "Mars", ["Venus", "Jupiter", "Mercury"]),
    ("Fact: The ocean between Europe and North America is the", ["Atlantic"],
     "Atlantic", ["Pacific", "Indian", "Arctic"]),
    ("Fact: The longest river in Egypt is the", ["Nile"],
     "Nile", ["Amazon", "Congo", "Niger"]),
    ("Fact: The flightless bird that lives in Antarctica is the", ["penguin"],
     "penguin", ["ostrich", "emu", "kiwi"]),
    ("Fact: The organ humans use to think is the", ["brain"],
     "brain", ["heart", "liver", "stomach"]),
    ("Fact: The white frozen precipitation that falls in winter is called",
     ["snow"], "snow", ["rain", "hail", "sleet"]),
    ("Fact: The planet famous for its large ring system is", ["Saturn"],
     "Saturn", ["Jupiter", "Uranus", "Neptune"]),
    ("Fact: The country famous for the Eiffel Tower is", ["France"],
     "France", ["Italy", "Spain", "Germany"]),
    ("Fact: The largest country in the world by land area is", ["Russia"],
     "Russia", ["Canada", "China", "Brazil"]),
    ("Fact: The yellow citrus fruit used to make lemonade is the", ["lemon"],
     "lemon", ["lime", "orange", "grapefruit"]),
]

# ------------------------------------------------------------- multihop (60)
# Distractor triples keyed by probe-swap item name; only items whose answer is
# an alphabetic content word (no digits, no number words, no 2-letter symbols)
# and whose prompt is unique. The paired swap_answer is included as a
# distractor wherever it is same-family.
MULTIHOP_DISTRACTORS = {
    "amazon-language": ["Spanish", "French", "Italian"],
    "animal-cover-turtle": ["scale", "fur", "feathers"],
    "animal-nose-elephant": ["tongue", "horn", "tail"],
    "atomic-80-state": ["solid", "gas", "plasma"],
    "beverage-source-wine": ["barley", "apple", "cherry"],
    "bird-color-swan": ["brown", "black", "gray"],
    "bird-country-eagle": ["England", "France", "Germany"],
    "bird-time-owl": ["day", "dawn", "noon"],
    "christmas-season": ["spring", "summer", "autumn"],
    "city-state-Philadelphia": ["Washington", "Virginia", "Ohio"],
    "colosseum-currency": ["peso", "franc", "yen"],
    "element-color-gold2": ["gray", "silver", "white"],
    "ex-city-capital-Barcelona-Toronto": ["Ottawa", "Rome", "Paris"],
    "ex-city-capital-Lyon-Naples": ["Rome", "Madrid", "Ottawa"],
    "ex-city-capital-Naples-Barcelona": ["Madrid", "Paris", "Berlin"],
    "ex-city-capital-Toronto-Lyon": ["Paris", "Rome", "Madrid"],
    "ex-city-continent-Toronto-Lyon": ["South", "Europe", "Asia"],
    "ex-city-currency-Toronto-Beijing": ["American", "Mexican", "British"],
    "ex-city-language-Lyon-Naples": ["Italian", "Spanish", "German"],
    "ex-element-state-26-8": ["liquid", "gas", "plasma"],
    "ex-element-state-8-26": ["solid", "liquid", "plasma"],
    "ex-planet-color-third-fourth": ["red", "green", "yellow"],
    "ex2-city-capital-Munich": ["Tokyo", "Paris", "Vienna"],
    "ex2-city-capital-Osaka": ["Berlin", "Seoul", "Beijing"],
    "ex2-city-continent-Lima": ["North", "Europe", "Africa"],
    "ex2-city-continent-Sydney": ["Africa", "Europe", "Asia"],
    "ex2-city-language-Cairo": ["Russian", "Turkish", "Hebrew"],
    "ex2-city-language-Moscow": ["Arabic", "Polish", "German"],
    "ex2-language-capital-Greek": ["Stockholm", "Budapest", "Warsaw"],
    "ex2-language-capital-Hungarian": ["Warsaw", "Athens", "Stockholm"],
    "ex2-language-capital-Polish": ["Athens", "Budapest", "Stockholm"],
    "ex2-language-capital-Swedish": ["Budapest", "Warsaw", "Athens"],
    "ex2-river-capital-Thames": ["Berlin", "Dublin", "Paris"],
    "food-animal-butter": ["bee", "goat", "sheep"],
    "food-animal-honey": ["cow", "wasp", "ant"],
    "fruit-grows-grape": ["tree", "bush", "stalk"],
    "gem-color-ruby": ["white", "green", "blue"],
    "gem-source-pearl": ["clam", "mussel", "snail"],
    "greatwall-ocean": ["Atlantic", "Indian", "Arctic"],
    "holiday-month-christmas2": ["April", "July", "October"],
    "instr-body-trumpet": ["hands", "fingers", "teeth"],
    "instr-hit-drums": ["keys", "bows", "picks"],
    "mars-color": ["blue", "green", "yellow"],
    "month-3-godof": ["love", "wisdom", "harvest"],
    "organ-acid-stomach": ["blood", "bile", "mucus"],
    "organ-location-brain": ["ribcage", "spine", "pelvis"],
    "organ-location-heart": ["head", "abdomen", "neck"],
    "osu-rival-mascot": ["badger", "spartan", "hawkeye"],
    "paper-continent": ["Europe", "Africa", "America"],
    "person-century-lincoln": ["Napoleonic", "Revolutionary", "Korean"],
    "person-country-napoleon": ["America", "England", "Spain"],
    "person-country-shakespeare": ["Germany", "France", "Italy"],
    "person-firstname-darwin": ["Albert", "Isaac", "William"],
    "person-firstname-einstein": ["Isaac", "Charles", "William"],
    "person-firstname-mozart": ["Charles", "Isaac", "Albert"],
    "person-firstname-newton": ["William", "Albert", "Charles"],
    "person-firstname-shakespeare": ["Wolfgang", "Isaac", "Albert"],
    "planet-rings-saturn2": ["spot", "moons", "clouds"],
    "rhyme-chair-flag": ["Wyoming", "Texas", "Montana"],
    "rhyme-rain-neighbor": ["France", "Spain", "Italy"],
    "rhyme-spoon-orbit": ["Jupiter", "Mars", "Venus"],
    "season-next-winter": ["autumn", "summer", "winter"],
    "spaceneedle-border": ["Mexico", "Oregon", "Idaho"],
    "sport-equip-tennis": ["club", "bat", "paddle"],
    "super-populous-capital": ["Delhi", "Tokyo", "Seoul"],
    "super-smallest-continent": ["Asia", "Africa", "America"],
    "tree-product-oak": ["cone", "berry", "walnut"],
    "vehicle-power-bicycle": ["arms", "feet", "hands"],
}

NUMBERISH = {"one", "two", "three", "four", "five", "six", "seven", "eight",
             "nine", "ten"}


def _gen_answers(ans):
    """Case variants for teacher-forced matching."""
    out = []
    for a in ([ans] if isinstance(ans, str) else ans):
        for v in (a, a.capitalize(), a.lower()):
            if v not in out:
                out.append(v)
    return out


def _fixed_prompt(stem, options):
    lines = [f"Question: {stem}"]
    lines += [f"{L}) {o}" for L, o in zip(LETTERS, options)]
    lines.append("Answer:")
    return "\n".join(lines)


def _emit(items, task, qid, gen_prompt, gen_answers, fixed_stem, answer_opt,
          distractors, rng):
    assert len(set([answer_opt] + distractors)) == 4, (task, qid)
    opts = [answer_opt] + list(distractors)
    rng.shuffle(opts)
    label = opts.index(answer_opt)
    answer_words = sorted({w.lower() for a in gen_answers + [answer_opt]
                           for w in a.split() if any(c.isalpha() for c in w)})
    items.append({
        "task": task, "fmt": "gen", "id": qid, "prompt": gen_prompt.rstrip(),
        "type": "gen", "answers": _gen_answers(gen_answers),
        "answer_words": answer_words,
    })
    items.append({
        "task": task, "fmt": "fixed", "id": qid,
        "prompt": _fixed_prompt(fixed_stem.rstrip().removeprefix("Fact: "), opts),
        "type": "logit", "option_strings": opts, "label": label,
        "answer_words": answer_words,
    })


def build_battery(cfg):
    n = cfg["n_items"]          # 60 underlying questions per task type
    seed = cfg["item_seed"]
    items = []

    # --- multihop: probe-swap items with content-word answers -------------
    raw = json.load(open(PROBE_SWAP))["items"]
    seen_prompts, pool = set(), []
    for it in sorted(raw, key=lambda x: x["name"]):
        if it["name"] not in MULTIHOP_DISTRACTORS:
            continue
        p = it["prompt"].rstrip()
        a = it["answer"]
        assert a.isalpha() and len(a) >= 3 and a.lower() not in NUMBERISH, it
        if p.lower() in seen_prompts:
            continue
        seen_prompts.add(p.lower())
        pool.append(it)
    assert len(pool) >= n, f"multihop pool {len(pool)} < {n}"
    rng = random.Random(seed)
    chosen = rng.sample(pool, n)
    for it in sorted(chosen, key=lambda x: x["name"]):
        _emit(items, "multihop", it["name"], it["prompt"], [it["answer"]],
              it["prompt"], it["answer"], MULTIHOP_DISTRACTORS[it["name"]],
              _rng(seed, "mh", it["name"]))
    # NOTE: per-item rng only shuffles option order; label balance is
    # checked (not forced) in the manifest.

    # --- singlehop: 30 capitals + 30 facts --------------------------------
    caps = CAPITALS[:30]
    all_caps = [c for _, c in caps]
    for i, (country, cap) in enumerate(caps):
        r = _rng(seed, "cap", i)
        distract = r.sample([c for c in all_caps if c != cap], 3)
        _emit(items, "singlehop", f"cap-{i}",
              f"Fact: The capital of {country} is", [cap],
              f"The capital of {country} is", cap, distract, r)
    for i, (prompt, answers, opt, distract) in enumerate(FACTS[:30]):
        r = _rng(seed, "fact", i)
        _emit(items, "singlehop", f"fact-{i}", prompt, answers,
              prompt, opt, distract, r)

    # --- rhyme ------------------------------------------------------------
    for i, (cue, defn, target, distract) in enumerate(RHYME[:n]):
        r = _rng(seed, "rhyme", i)
        stem = f"What word rhymes with {cue} and means {defn}?"
        gen_prompt = f"Q: {stem}\nA: The word is"
        gen_answers = [target, target.capitalize(), f'"{target}', f"**{target}",
                       f'"{target.capitalize()}', f"**{target.capitalize()}"]
        assert len(set([target] + distract)) == 4, (cue, target, distract)
        opts = [target] + list(distract)
        r.shuffle(opts)
        label = opts.index(target)
        answer_words = [target.lower()]
        items.append({
            "task": "rhyme", "fmt": "gen", "id": f"rhyme-{i}",
            "prompt": gen_prompt, "type": "gen", "answers": gen_answers,
            "answer_words": answer_words,
        })
        items.append({
            "task": "rhyme", "fmt": "fixed", "id": f"rhyme-{i}",
            "prompt": _fixed_prompt(stem, opts), "type": "logit",
            "option_strings": opts, "label": label,
            "answer_words": answer_words,
        })

    counts = {}
    for it in items:
        counts[(it["task"], it["fmt"])] = counts.get((it["task"], it["fmt"]), 0) + 1
    for (t, f), c in counts.items():
        assert c == n, f"{t}/{f}: {c} != {n}"
    return items

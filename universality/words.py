COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "silver", "gold", "violet", "crimson", "beige",
]

ANIMALS = [
    "dog", "cat", "horse", "cow", "pig", "sheep", "goat", "chicken", "duck",
    "rabbit", "mouse", "rat", "lion", "tiger", "bear", "wolf", "fox", "deer",
    "elephant", "monkey", "snake", "frog", "fish", "shark", "whale", "dolphin",
    "eagle", "owl", "crow", "sparrow", "spider", "ant", "bee", "butterfly",
    "camel", "donkey", "squirrel", "turtle", "penguin", "kangaroo", "zebra",
    "giraffe", "leopard", "panda", "otter", "seal", "crab", "lobster", "octopus",
    "goose", "swan", "pigeon", "parrot", "hawk", "falcon", "bat", "moth",
]

BODY = [
    "head", "face", "eye", "ear", "nose", "mouth", "tooth", "tongue", "neck",
    "shoulder", "arm", "elbow", "wrist", "hand", "finger", "thumb", "chest",
    "heart", "lung", "stomach", "back", "hip", "leg", "knee", "ankle", "foot",
    "toe", "skin", "bone", "blood", "brain", "hair", "throat", "spine",
]

CLOTHES = [
    "shirt", "pants", "dress", "skirt", "coat", "jacket", "sweater", "hat",
    "cap", "scarf", "glove", "sock", "shoe", "boot", "belt", "tie", "suit",
    "vest", "collar", "sleeve", "pocket", "button", "zipper",
]

FOOD = [
    "bread", "butter", "cheese", "milk", "egg", "meat", "beef", "pork",
    "rice", "pasta", "soup", "salad", "apple", "banana", "orange", "grape",
    "lemon", "peach", "pear", "cherry", "melon", "potato", "tomato", "onion",
    "carrot", "pepper", "garlic", "salt", "sugar", "honey", "chocolate",
    "cake", "cookie", "pie", "tea", "coffee", "wine", "beer", "juice",
    "water", "corn", "bean", "nut", "mushroom", "fish", "chicken",
]

NATURE = [
    "sun", "moon", "star", "sky", "cloud", "rain", "snow", "wind", "storm",
    "thunder", "lightning", "fog", "ice", "fire", "smoke", "ash", "mountain",
    "hill", "valley", "river", "lake", "ocean", "sea", "beach", "island",
    "forest", "tree", "leaf", "branch", "root", "flower", "grass", "rock",
    "stone", "sand", "soil", "mud", "cave", "desert", "field", "meadow",
    "wave", "tide", "volcano", "glacier", "canyon", "cliff", "swamp",
]

HOUSEHOLD = [
    "table", "chair", "bed", "sofa", "desk", "lamp", "mirror", "clock",
    "door", "window", "wall", "floor", "ceiling", "roof", "stairs", "kitchen",
    "oven", "stove", "fridge", "sink", "cup", "glass", "plate", "bowl",
    "spoon", "fork", "knife", "pot", "pan", "kettle", "towel", "blanket",
    "pillow", "curtain", "carpet", "shelf", "drawer", "cabinet", "bucket",
    "broom", "candle", "vase", "basket", "hammer", "nail", "screw", "rope",
    "ladder", "key", "lock", "chain", "wire", "pipe", "brick", "glue",
]

OBJECTS = [
    "book", "pen", "pencil", "paper", "letter", "card", "map", "photo",
    "camera", "phone", "computer", "screen", "keyboard", "radio", "television",
    "car", "truck", "bus", "train", "plane", "boat", "ship", "bicycle",
    "wheel", "engine", "road", "bridge", "tunnel", "station", "airport",
    "money", "coin", "wallet", "bag", "box", "bottle", "jar", "can",
    "ball", "toy", "doll", "kite", "drum", "guitar", "piano", "violin",
    "flute", "bell", "sword", "shield", "arrow", "bow", "gun", "bomb",
    "flag", "crown", "ring", "necklace", "watch", "umbrella", "tent",
    "needle", "thread", "web", "net", "hook", "anchor", "sail", "oar",
]

PLACES_PEOPLE = [
    "house", "school", "church", "hospital", "library", "museum", "theater",
    "restaurant", "hotel", "shop", "market", "farm", "factory", "office",
    "prison", "castle", "palace", "tower", "garden", "park", "zoo", "city",
    "town", "village", "street", "doctor", "nurse", "teacher", "student",
    "farmer", "soldier", "sailor", "pilot", "driver", "cook", "baker",
    "hunter", "king", "queen", "prince", "princess", "knight", "priest",
    "judge", "lawyer", "artist", "singer", "dancer", "writer", "poet",
    "mother", "father", "sister", "brother", "uncle", "aunt", "cousin",
    "baby", "child", "boy", "girl", "man", "woman", "friend", "neighbor",
]


def candidate_pool(countries):
    pool = (COLORS + ANIMALS + BODY + CLOTHES + FOOD + NATURE + HOUSEHOLD
            + OBJECTS + PLACES_PEOPLE + list(countries))
    seen, out = set(), []
    for w in pool:
        if w.lower() not in seen:
            seen.add(w.lower())
            out.append(w)
    return out

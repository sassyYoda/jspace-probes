"""Deterministic 8-step evidence ladders: step 1 = oblique base, steps 2-7 add one
hint each (cumulative), step 8 names the concept. Concept token must not appear in
steps 1-7 (asserted token-level at runtime)."""

COUNTRY_BASE = "The place is nice."
NOUN_BASE = "The thing is interesting."

COUNTRIES = {
    "France": [
        "It is a country.",
        "It is in western Europe.",
        "It is known for its wine, cheese and fashion houses.",
        "The Eiffel Tower stands in its capital city.",
        "Its capital is Paris, on the river Seine.",
        "Its people speak French.",
    ],
    "Germany": [
        "It is a country.",
        "It is in central Europe.",
        "It is known for beer, sausages and precision engineering.",
        "Carmakers like BMW and Volkswagen are based there.",
        "Its capital is Berlin.",
        "Its people speak German.",
    ],
    "Spain": [
        "It is a country.",
        "It is in southern Europe.",
        "It is known for flamenco, tapas and bullfighting.",
        "It occupies most of the Iberian Peninsula.",
        "Its capital is Madrid.",
        "Its people speak Spanish.",
    ],
    "Italy": [
        "It is a country.",
        "It is in southern Europe.",
        "It is known for pasta, pizza and opera.",
        "It is shaped like a boot on the map.",
        "Its capital is Rome, home of the Colosseum.",
        "Its people speak Italian.",
    ],
    "Japan": [
        "It is a country.",
        "It is an island nation in East Asia.",
        "It is known for sushi, anime and cherry blossoms.",
        "Mount Fuji is its most famous peak.",
        "Its capital is Tokyo.",
        "Its people speak Japanese.",
    ],
    "Brazil": [
        "It is a country.",
        "It is in South America.",
        "It is known for carnival, samba and football.",
        "A giant statue of Christ overlooks Rio de Janeiro.",
        "Its capital is Brasilia.",
        "Its people speak Portuguese.",
    ],
    "China": [
        "It is a country.",
        "It is in East Asia.",
        "It is known for tea, silk and dumplings.",
        "A great wall thousands of miles long crosses its north.",
        "Its capital is Beijing.",
        "Its people speak Mandarin.",
    ],
    "Russia": [
        "It is a country.",
        "It spans eastern Europe and northern Asia.",
        "It is known for ballet, vodka and long winters.",
        "The Kremlin and Red Square stand in its capital.",
        "Its capital is Moscow.",
        "Its people speak Russian.",
    ],
    "Egypt": [
        "It is a country.",
        "It is in northeastern Africa.",
        "It is known for ancient temples and desert sands.",
        "Great pyramids and a sphinx stand near its capital.",
        "Its capital is Cairo.",
        "Its people once wrote in hieroglyphs.",
    ],
    "India": [
        "It is a country.",
        "It is in South Asia.",
        "It is known for curry, cricket and Bollywood films.",
        "The Taj Mahal is its most famous monument.",
        "Its capital is New Delhi.",
        "Its people speak Hindi among many languages.",
    ],
    "Canada": [
        "It is a country.",
        "It is in North America.",
        "It is known for maple syrup, ice hockey and politeness.",
        "It stretches from the Atlantic to the Pacific above the United States.",
        "Its capital is Ottawa.",
        "A red maple leaf appears on its flag.",
    ],
    "Mexico": [
        "It is a country.",
        "It is in North America, just south of the United States.",
        "It is known for tacos, mariachi and ancient ruins.",
        "Aztec and Maya civilizations flourished there.",
        "Its capital is a huge city built on a drained lake.",
        "Its people speak Spanish and celebrate the Day of the Dead.",
    ],
}

NOUNS = {
    "dog": [
        "It is a four-legged animal.",
        "It is a common household pet.",
        "It barks at strangers and wags its tail.",
        "It loves to fetch sticks and chew bones.",
        "It is called man's best friend.",
        "Its young are called puppies.",
    ],
    "cat": [
        "It is a small four-legged animal.",
        "It is a common household pet.",
        "It purrs when content and hisses when angry.",
        "It chases mice and climbs curtains.",
        "It is said to have nine lives.",
        "Its young are called kittens.",
    ],
    "spider": [
        "It is a small creature.",
        "It has eight legs.",
        "It spins silk to catch insects.",
        "Its web glistens with dew in the morning.",
        "Many people fear it, a fear called arachnophobia.",
        "Famous kinds include tarantulas and black widows.",
    ],
    "elephant": [
        "It is a very large animal.",
        "It lives in Africa and Asia.",
        "It has thick grey skin and big ears.",
        "It uses its long trunk to grab food and spray water.",
        "Its ivory tusks made it a target for poachers.",
        "It is the largest land animal on Earth.",
    ],
    "horse": [
        "It is a large four-legged animal.",
        "People have ridden it for thousands of years.",
        "It gallops, trots and neighs.",
        "It wears iron shoes and eats hay and oats.",
        "Jockeys race them at the derby.",
        "Its young are called foals.",
    ],
    "lion": [
        "It is a large wild animal.",
        "It is a big cat living on the African savanna.",
        "The males have thick manes.",
        "It hunts in groups called prides.",
        "Its roar can be heard miles away.",
        "It is called the king of the jungle.",
    ],
    "shark": [
        "It is a large animal that lives in the sea.",
        "It has existed since before the dinosaurs.",
        "It has rows of razor-sharp teeth that regrow.",
        "Its dorsal fin cutting the surface frightens swimmers.",
        "The great white is its most feared kind.",
        "The movie Jaws made it famous.",
    ],
    "eagle": [
        "It is a bird.",
        "It is a large bird of prey.",
        "It soars high and spots prey with sharp eyes.",
        "It nests on cliffs and mountain tops.",
        "The bald variety symbolizes the United States.",
        "It grips its prey with powerful talons.",
    ],
    "bee": [
        "It is a small insect.",
        "It buzzes from flower to flower.",
        "It collects nectar and pollen.",
        "It lives in hives ruled by a queen.",
        "It makes honey and wax.",
        "It stings once and then dies.",
    ],
    "snake": [
        "It is an animal without legs.",
        "It slithers along the ground.",
        "It flicks a forked tongue and sheds its skin.",
        "Some kinds have deadly venom in their fangs.",
        "Cobras and pythons are famous kinds.",
        "It hisses when threatened.",
    ],
    "piano": [
        "It is a large object found in many homes.",
        "It is a musical instrument.",
        "It has a keyboard of black and white keys.",
        "Hammers inside strike strings when keys are pressed.",
        "Concert halls feature grand ones.",
        "Beethoven and Chopin wrote music for it.",
    ],
    "guitar": [
        "It is an object many teenagers want.",
        "It is a musical instrument with strings.",
        "It has six strings and a long fretted neck.",
        "You strum or pluck it, sometimes with a pick.",
        "Rock bands rely on the electric kind.",
        "Jimi Hendrix played it like no one else.",
    ],
    "clock": [
        "It is a common household object.",
        "It hangs on walls or stands on towers.",
        "It has a face with moving hands.",
        "It ticks steadily and chimes on the hour.",
        "Big Ben in London is a giant famous one.",
        "People check it to tell the time.",
    ],
    "mirror": [
        "It is a common household object.",
        "It hangs on walls and sits in bathrooms.",
        "It is made of glass with a silvered back.",
        "Breaking one is said to bring seven years of bad luck.",
        "The evil queen in Snow White consults a magic one.",
        "You see your own reflection in it.",
    ],
    "candle": [
        "It is a small household object.",
        "It is made of wax with a wick through the middle.",
        "You light it when the power goes out.",
        "It burns with a small flickering flame.",
        "Birthday cakes carry them, one per year.",
        "You blow them out and make a wish.",
    ],
    "umbrella": [
        "It is an object people carry.",
        "You take it along when the sky looks grey.",
        "It folds up and opens with a click.",
        "Its canopy stretches over metal ribs on a pole.",
        "It keeps the rain off your head.",
        "Mary Poppins flies holding one.",
    ],
    "bicycle": [
        "It is a machine people use every day.",
        "It has two wheels and a frame.",
        "You pedal it and steer with handlebars.",
        "It has a chain, a saddle and a bell.",
        "Children learn to ride it with training wheels.",
        "The Tour de France is raced on them.",
    ],
    "train": [
        "It is a large machine for travel.",
        "It runs on steel rails.",
        "It has many cars pulled along a track.",
        "It stops at platforms and stations.",
        "Steam ones once crossed continents whistling.",
        "Passengers board it at the railway platform.",
    ],
    "boat": [
        "It is a vehicle of sorts.",
        "It travels on water.",
        "It floats and is steered with a rudder.",
        "Some have sails, others have motors or oars.",
        "Fishermen take theirs out at dawn.",
        "You row it gently down the stream.",
    ],
    "bread": [
        "It is something found in every kitchen.",
        "It is a basic food eaten daily around the world.",
        "It is baked from flour, water and yeast.",
        "Its crust is crisp and its inside is soft.",
        "You slice it and toast it for breakfast.",
        "Bakers pull fresh loaves of it from the oven.",
    ],
    "cheese": [
        "It is something found in most kitchens.",
        "It is a food made from milk.",
        "It is aged in wheels and blocks.",
        "Mice are said to love it.",
        "Cheddar, brie and gouda are kinds of it.",
        "It melts deliciously on pizza.",
    ],
    "apple": [
        "It is something you can eat.",
        "It is a fruit that grows on trees.",
        "It is crisp and can be red or green.",
        "One a day is said to keep the doctor away.",
        "Legend says one fell on Newton's head.",
        "You can bake it into a pie with cinnamon.",
    ],
    "coffee": [
        "It is something many people consume daily.",
        "It is a hot drink.",
        "It is brewed from roasted ground beans.",
        "Its caffeine wakes people up in the morning.",
        "Espresso and cappuccino are made from it.",
        "Baristas serve it in cafes.",
    ],
    "hammer": [
        "It is a common tool.",
        "It is found in every toolbox.",
        "It has a heavy metal head on a wooden handle.",
        "Carpenters swing it all day.",
        "You drive nails into wood with it.",
        "The claw side pulls nails back out.",
    ],
    "knife": [
        "It is a common tool.",
        "It is found in every kitchen drawer.",
        "It has a sharp blade and a handle.",
        "Chefs keep theirs razor sharp.",
        "You chop vegetables and slice meat with it.",
        "It comes with the fork and spoon in a table setting.",
    ],
    "key": [
        "It is a small metal object.",
        "It fits in your pocket or on a ring.",
        "You turn it in a lock.",
        "It opens doors and starts cars.",
        "Losing it leaves you stuck outside.",
        "A locksmith cuts copies of it.",
    ],
    "moon": [
        "It is something everyone has seen.",
        "It appears in the night sky.",
        "It waxes and wanes through phases each month.",
        "Its pull causes the ocean tides.",
        "Astronauts planted a flag on it in 1969.",
        "It is Earth's only natural satellite.",
    ],
    "sun": [
        "It is something everyone has seen.",
        "It appears in the sky every day.",
        "It rises in the east and sets in the west.",
        "Plants turn toward its light to grow.",
        "It is the star at the center of our solar system.",
        "Its rays can burn your skin at noon.",
    ],
}

PROBES = ["volcano", "wizard", "trumpet", "galaxy", "castle"]


def concepts():
    out = {}
    for name, hints in COUNTRIES.items():
        out[name] = {"kind": "country", "base": COUNTRY_BASE, "hints": hints,
                     "explicit": f"{name} is the country in question."}
    for name, hints in NOUNS.items():
        out[name] = {"kind": "noun", "base": NOUN_BASE, "hints": hints,
                     "explicit": f"The {name} is the thing in question."}
    return out


def ladder(spec):
    steps = [spec["base"]]
    for i in range(len(spec["hints"])):
        steps.append(" ".join([spec["base"]] + spec["hints"][: i + 1]))
    steps.append(steps[-1] + " " + spec["explicit"])
    return [s.rstrip() for s in steps]

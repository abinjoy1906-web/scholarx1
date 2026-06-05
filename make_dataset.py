"""
make_dataset.py
---------------
Generates a synthetic labelled text dataset that mirrors the structure
of the 20 Newsgroups collection. Used as an offline fallback.

Produces ~3 200 realistic short documents across 8 categories and
saves them to data/newsgroups_synthetic.csv.
"""

import os
import random
import csv

random.seed(42)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
OUT_PATH  = os.path.join(DATA_DIR, "newsgroups_synthetic.csv")

# ── Templates per category ────────────────────────────────────────────────────

TEMPLATES = {
    "Hockey": [
        "The {team1} defeated the {team2} in a thrilling overtime game last night. {player} scored the winning goal.",
        "{player} recorded a hat-trick as the {team1} crushed the {team2} by {score}. The crowd went wild.",
        "The NHL playoffs are heating up. {team1} faces {team2} in a pivotal game seven showdown.",
        "Goaltender {goalie} made {saves} saves to preserve the shutout for {team1}.",
        "Trade rumours suggest {player} may be moving to {team2} before the deadline.",
        "The power-play unit for {team1} has been unstoppable, converting at {pct}% this season.",
        "{team1} coach announced lineup changes ahead of the crucial playoff matchup against {team2}.",
        "After a slow start, {player} has found his form and now leads the league in points.",
        "The ice hockey world cup qualifier ended with {country1} beating {country2} in penalties.",
        "Analysts predict {team1} will win the Stanley Cup this year based on their recent run of form.",
    ],
    "Baseball": [
        "The {team1} rallied from five runs down to edge out {team2} in extra innings.",
        "{pitcher} threw a complete-game shutout, striking out {k} batters against {team2}.",
        "{player} hit his {n}th home run of the season, putting {team1} ahead in the eighth.",
        "The MLB draft saw {team1} select shortstop {prospect} with the first overall pick.",
        "Spring training opens next week; {team1} manager expects a strong season.",
        "{player}'s batting average has improved to .{avg} after a slow start to the season.",
        "The {team1} bullpen has been shaky, blowing three saves in the last two weeks.",
        "World Series preview: {team1} vs {team2} — who has the edge on the mound?",
        "Rain delays pushed the {team1} vs {team2} game to a double-header tomorrow.",
        "Veteran outfielder {player} announced his retirement after {years} seasons in the majors.",
    ],
    "Medicine": [
        "Researchers at {uni} published findings showing {drug} reduces {condition} risk by {pct}%.",
        "A new clinical trial for {disease} treatment shows promising results in phase two testing.",
        "Doctors recommend {vaccine} booster doses for adults over {age} amid rising {disease} cases.",
        "Scientists have identified a genetic marker linked to {condition} susceptibility.",
        "The FDA approved {drug} for the treatment of {disease} following successful trials.",
        "{pct}% of patients in the study showed complete remission after {weeks} weeks of {therapy}.",
        "A meta-analysis of {n} studies confirms that {lifestyle} reduces {condition} mortality.",
        "Hospitals report a surge in {disease} admissions during the current {season} season.",
        "Neuroscientists at {uni} have mapped new neural pathways involved in {function}.",
        "A breakthrough mRNA therapy targets {gene} mutations associated with {cancer} cancer.",
    ],
    "Space": [
        "NASA confirmed that the {mission} spacecraft successfully entered orbit around {planet}.",
        "Astronomers discovered a new exoplanet in the habitable zone of star {star}.",
        "The {rocket} launch from Cape Canaveral placed {satellite} into geostationary orbit.",
        "Images from the James Webb Space Telescope reveal unprecedented detail in {nebula}.",
        "SpaceX plans to launch its {spacecraft} on a crewed mission to the ISS next month.",
        "Researchers detected unusual radio signals from {galaxy} that may indicate stellar activity.",
        "Mars rover {rover} collected rock samples that hint at ancient microbial life.",
        "A {size}-metre asteroid will pass within {dist} km of Earth on {date}.",
        "ESA's {probe} has completed its flyby of {comet} and transmitted high-resolution data.",
        "The Artemis programme aims to return astronauts to the Moon's surface by {year}.",
    ],
    "Politics": [
        "Lawmakers debated the {bill} act, which would tighten background check requirements.",
        "The Senate voted {yea}-{nay} to advance the {bill} bill on firearm regulation.",
        "Governor {name} signed an executive order expanding {policy} rights in the state.",
        "Protestors gathered outside the {building} demanding action on {issue}.",
        "The Supreme Court agreed to hear arguments on the constitutionality of {law}.",
        "Poll shows {pct}% of voters support stricter {policy} legislation.",
        "Advocacy groups filed a lawsuit challenging the {state} {law} law in federal court.",
        "Presidential candidates clashed on {topic} during last night's televised debate.",
        "Congress is deadlocked over the {bill} amendment, delaying the budget vote.",
        "Former Secretary {name} called for bipartisan cooperation on {issue} reform.",
    ],
    "Religion": [
        "Religious leaders gathered in {city} to discuss interfaith dialogue and peace.",
        "The {religion} community celebrated {festival} with prayers and community gatherings.",
        "Theological scholars debate the interpretation of {text} in modern contexts.",
        "A {religion} temple in {city} was inaugurated, welcoming thousands of devotees.",
        "Pilgrims from {country} arrived in {holy_city} for the annual {pilgrimage}.",
        "The role of {religion} in public life was the subject of a {uni} symposium.",
        "A new translation of the {text} has sparked debate among {religion} academics.",
        "{religion} missionaries established schools and clinics in {region}.",
        "The {denomination} council released a statement on {social_issue} and faith.",
        "Young people are increasingly exploring {religion} spirituality outside formal institutions.",
    ],
    "Computing": [
        "The new {gpu} graphics card achieves {fps} FPS in 4K ray-traced rendering benchmarks.",
        "Developers released {engine} version {ver}, introducing real-time global illumination.",
        "A {renderer} renderer algorithm reduces aliasing artefacts in real-time 3D scenes.",
        "OpenGL {ver} introduces compute shaders that accelerate {effect} rendering.",
        "Research presents a novel denoising approach for path-traced {scene} scenes.",
        "The {software} update adds support for {format} texture compression on mobile GPUs.",
        "Benchmarks show the {cpu} improves {task} performance by {pct}% over the previous gen.",
        "A new open-source library simplifies {effect} shading for game developers.",
        "Vulkan API adoption is growing as studios migrate from {api} for better GPU control.",
        "The {studio} demo showcased photorealistic {scene} rendering at interactive frame rates.",
    ],
    "Christianity": [
        "The {denomination} church released a pastoral letter on {social_issue} and scripture.",
        "Archbishop {name} delivered a homily on {theme} at the {cathedral} Cathedral.",
        "Thousands of pilgrims gathered at {shrine} for the feast of {saint}.",
        "Theological colleges report growing enrolment in {programme} ministry programmes.",
        "A new translation of the {text} aims to make scripture accessible to young readers.",
        "The {council} council affirmed traditional teachings on {doctrine} at its annual meeting.",
        "Christian aid organisations are providing relief to {region} disaster victims.",
        "A panel of {denomination} bishops discussed ecumenical relations at the {conference}.",
        "Easter celebrations drew record attendance at {church} churches across {country}.",
        "Pope {name} called for global solidarity and prayer for peace in {region}.",
    ],
}

# ── Filler word banks ─────────────────────────────────────────────────────────
FILLERS = {
    "team1":      ["Maple Leafs", "Bruins", "Penguins", "Rangers", "Red Wings",
                   "Yankees", "Red Sox", "Dodgers", "Cubs", "Astros"],
    "team2":      ["Canadiens", "Blackhawks", "Oilers", "Flames", "Senators",
                   "Mets", "Giants", "Padres", "Cardinals", "Braves"],
    "player":     ["Connor McDavid", "Sidney Crosby", "Alex Ovechkin", "Auston Matthews",
                   "Mike Trout", "Shohei Ohtani", "Aaron Judge", "Freddie Freeman"],
    "goalie":     ["Andrei Vasilevskiy", "Carey Price", "Marc-André Fleury", "Tuukka Rask"],
    "pitcher":    ["Gerrit Cole", "Jacob deGrom", "Max Scherzer", "Clayton Kershaw"],
    "prospect":   ["Jackson Holliday", "Paul Skenes", "Dylan Crews"],
    "country1":   ["Canada", "USA", "Russia", "Sweden", "Finland", "Czech Republic"],
    "country2":   ["Germany", "Slovakia", "Switzerland", "Latvia", "Denmark"],
    "score":      ["5-1", "4-2", "6-3", "3-0", "7-4"],
    "pct":        ["12", "18", "24", "31", "42", "55", "67", "73", "88"],
    "saves":      ["34", "38", "42", "29", "51"],
    "k":          ["9", "11", "13", "15", "7"],
    "n":          ["10", "15", "22", "30", "40"],
    "avg":        ["285", "312", "298", "340", "270"],
    "years":      ["12", "15", "18", "22"],
    "uni":        ["Harvard", "Stanford", "MIT", "Johns Hopkins", "Oxford", "Cambridge"],
    "drug":       ["remdesivir", "metformin", "atorvastatin", "pembrolizumab", "rituximab"],
    "disease":    ["diabetes", "Alzheimer's", "hypertension", "COVID-19", "influenza"],
    "condition":  ["cardiovascular", "metabolic", "neurological", "autoimmune"],
    "cancer":     ["breast", "lung", "colon", "prostate", "pancreatic"],
    "gene":       ["BRCA1", "TP53", "KRAS", "EGFR", "ALK"],
    "vaccine":    ["flu", "HPV", "pneumococcal", "shingles"],
    "therapy":    ["immunotherapy", "chemotherapy", "radiation", "gene therapy"],
    "lifestyle":  ["regular exercise", "Mediterranean diet", "stress reduction"],
    "function":   ["memory consolidation", "pain perception", "emotional regulation"],
    "season":     ["winter", "spring", "summer", "autumn"],
    "age":        ["50", "60", "65", "70"],
    "weeks":      ["8", "12", "16", "24"],
    "mission":    ["Artemis III", "Perseverance", "Voyager", "Cassini", "Juno"],
    "planet":     ["Mars", "Jupiter", "Saturn", "Venus", "Europa"],
    "star":       ["Kepler-442", "TRAPPIST-1", "Proxima Centauri", "Tau Ceti"],
    "rocket":     ["Falcon 9", "SLS", "Ariane 6", "Vulcan Centaur"],
    "satellite":  ["Starlink cluster", "GPS III", "Intelsat 40E", "GOES-18"],
    "nebula":     ["Orion Nebula", "Crab Nebula", "Eagle Nebula", "Helix Nebula"],
    "spacecraft": ["Crew Dragon", "Starliner", "Orion"],
    "galaxy":     ["Andromeda", "Triangulum", "Sombrero", "Whirlpool"],
    "rover":      ["Perseverance", "Curiosity", "Ingenuity"],
    "size":       ["50", "120", "300", "800"],
    "dist":       ["450 000", "1.2 million", "3 million"],
    "date":       ["November 14", "March 22", "July 3", "September 9"],
    "probe":      ["Rosetta", "Hayabusa2", "OSIRIS-REx"],
    "comet":      ["67P", "Halley", "Churyumov-Gerasimenko"],
    "year":       ["2026", "2027", "2028", "2030"],
    "bill":       ["Safe Carry", "Firearm Safety", "Second Amendment Preservation", "Universal Background Check"],
    "yea":        ["52", "58", "61", "49"],
    "nay":        ["47", "41", "38", "50"],
    "name":       ["Johnson", "Williams", "Rodriguez", "Smith", "Thompson"],
    "policy":     ["gun-ownership", "concealed carry", "open-carry", "firearm storage"],
    "building":   ["Capitol", "Statehouse", "City Hall", "Supreme Court"],
    "issue":      ["gun violence", "background checks", "assault weapons", "red-flag laws"],
    "law":        ["Firearm Safety Act", "Open Carry Law", "Red Flag Law", "Background Check Act"],
    "state":      ["Texas", "California", "Florida", "New York", "Colorado"],
    "topic":      ["gun control", "border security", "healthcare", "climate change"],
    "religion":   ["Buddhist", "Hindu", "Muslim", "Jewish", "Sikh", "Taoist"],
    "festival":   ["Diwali", "Eid al-Fitr", "Hanukkah", "Vesak", "Vaisakhi"],
    "text":       ["Quran", "Torah", "Bhagavad Gita", "Tripitaka", "Guru Granth Sahib"],
    "holy_city":  ["Mecca", "Jerusalem", "Varanasi", "Amritsar", "Bodh Gaya"],
    "pilgrimage": ["Hajj", "Camino de Santiago", "Kumbh Mela", "Way of the Cross"],
    "social_issue":["climate change", "poverty", "immigration", "family ethics"],
    "denomination":["Baptist", "Methodist", "Lutheran", "Anglican", "Presbyterian"],
    "region":     ["the Middle East", "sub-Saharan Africa", "Southeast Asia", "Eastern Europe"],
    "gpu":        ["RTX 5080", "RX 9700 XT", "Arc B770", "RTX 5090"],
    "fps":        ["120", "85", "144", "60"],
    "engine":     ["Unreal Engine", "Unity", "Godot", "CryEngine"],
    "ver":        ["5.4", "2.0", "4.6", "3.2"],
    "renderer":   ["path-tracing", "rasterisation", "hybrid ray-tracing", "radiosity"],
    "effect":     ["ambient occlusion", "volumetric lighting", "subsurface scattering", "bloom"],
    "scene":      ["forest", "urban", "underwater", "space", "interior"],
    "software":   ["DirectX 13", "Metal 3", "Vulkan 1.4", "OpenGL 4.7"],
    "format":     ["ASTC", "BC7", "ETC2", "PVRTC"],
    "cpu":        ["Ryzen 9 9950X", "Core Ultra 9 285K", "M4 Pro"],
    "task":       ["shader compilation", "physics simulation", "AI inference"],
    "api":        ["DirectX 11", "OpenGL", "DirectX 12"],
    "studio":     ["Epic Games", "DICE", "Naughty Dog", "id Software"],
    "cathedral":  ["Westminster", "Notre-Dame", "St Patrick's", "Cologne"],
    "shrine":     ["Lourdes", "Fatima", "Medjugorje", "Czestochowa"],
    "saint":      ["Francis of Assisi", "Thérèse of Lisieux", "John Paul II", "Mary Magdalene"],
    "programme":  ["pastoral", "youth", "missionary", "chaplaincy"],
    "council":    ["Vatican", "Anglican", "Synodal", "Evangelical"],
    "conference": ["World Council of Churches", "Lausanne Conference", "Lambeth Conference"],
    "church":     ["Catholic", "Protestant", "Orthodox", "Evangelical"],
    "country":    ["Ireland", "Poland", "Brazil", "Philippines", "Italy"],
    "doctrine":   ["marriage", "the afterlife", "salvation", "social justice"],
    "theme":      ["mercy", "forgiveness", "hope", "solidarity", "reconciliation"],
}


def fill(template: str) -> str:
    """Replace every {placeholder} in template with a random value."""
    result = template
    for key, choices in FILLERS.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, random.choice(choices), 1)
    return result


def generate_dataset(n_per_class: int = 400) -> list:
    """Generate n_per_class documents for each of the 8 categories.

    Args:
        n_per_class: Documents to synthesise per category.

    Returns:
        List of (text, label) tuples.
    """
    rows = []
    for label, templates in TEMPLATES.items():
        for _ in range(n_per_class):
            tmpl = random.choice(templates)
            text = fill(tmpl)
            # Append a second random sentence for variety
            tmpl2 = random.choice(templates)
            text += " " + fill(tmpl2)
            rows.append((text, label))
    random.shuffle(rows)
    return rows


def save_dataset(rows: list, path: str):
    """Write rows to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"💾  Saved {len(rows)} documents → {path}")


if __name__ == "__main__":
    rows = generate_dataset(n_per_class=400)
    save_dataset(rows, OUT_PATH)

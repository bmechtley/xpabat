# ─────────────────────────────────────────────
# Species
# ─────────────────────────────────────────────
PROFILES = [
    # ── prior: relative weight for this site (bridge colony, Tucson AZ, late May)
    # Acoustic criteria alone are scored 0–1; the prior multiplies that score so
    # ties resolve in favour of the locally most likely species.  Values > 1 boost,
    # values < 1 suppress.  A species still needs ≥ 0.5 raw acoustic score to win.
    {
        "name": "Tadarida brasiliensis", "short": "TABR",
        "Fchar": (20, 27), "Fmin": (18, 25), "dur": (8, 25), "sweep": (0.1, 1.2),
        "bw": (1, 16), "cf_frac": (0.45, 1.0),   # narrow band, mostly CF
        "prior": 1.6,   # dominant bridge-colony bat in Tucson; very likely majority of calls
        "common": "Mexican Free-tailed Bat",
        "call_type": "Nearly constant-frequency (CF), extremely narrow bandwidth. Characteristic search-phase call at ~20–26 kHz. Molossid — not a vespertilionid.",
        "desc": (
            "Most abundant bat in North America and almost certainly the dominant species here. "
            "The Campbell Ave bridge hosts a large colony. Forms the largest mammal aggregations "
            "on earth (Bracken Cave, TX: ~15 million). Long narrow wings for fast, high flight. "
            "Narrow-band CF call is highly distinctive and rarely confused with other western NA species. "
            "Year-round resident in Tucson; colony emerges at dusk in a spectacular column."
        ),
        "habitat": "Open habitats — agricultural, suburban, over water. Roosts in vast cave colonies, buildings, and bridges. The Rillito corridor is prime foraging habitat.",
        "range": "Southern US through Central America and most of South America. Year-round resident in warmest parts of range including Tucson.",
        "ipi_ms": "50–80",
        "refs": [
            ["Williams et al. (1973) Anim Behav 21:302–321",
             "https://scholar.google.com/scholar?q=Williams+1973+bat+pursuit+echolocation+Animal+Behaviour"],
            ["Simmons & Stein (1980) J Comp Physiol 135:335–353",
             "https://scholar.google.com/scholar?q=Simmons+Stein+1980+acoustic+interference+bat+echolocation+Journal+Comparative+Physiology"],
            ["O'Shea & Bogan (2003) Monitoring Trends in Bat Populations of the US and Territories, USGS",
             "https://scholar.google.com/scholar?q=O%27Shea+Bogan+2003+monitoring+trends+bat+populations+United+States+territories+USGS"],
        ],
    },
    {
        "name": "Eptesicus fuscus", "short": "EPFU",
        "Fchar": (24, 35), "Fmin": (20, 28), "dur": (8, 20), "sweep": (0.5, 3.5),
        "bw": (6, 22), "cf_frac": (0.10, 0.70),  # moderate FM sweep
        "prior": 1.0,
        "common": "Big Brown Bat",
        "call_type": "Shallow FM sweep with quasi-CF tail. Relatively low frequency for a vespertilionid; characteristic frequency slightly higher in SW populations (~25–33 kHz) than in the east.",
        "desc": (
            "Common, year-round urban bat in Tucson. Large body size (~15–20 g), slow powerful flight. "
            "Roosts in buildings and bridges year-round. Calls overlap in frequency with TABR "
            "but are broader-band and have a more pronounced FM component. "
            "Likely present in small numbers alongside the TABR colony."
        ),
        "habitat": "Highly adaptable — buildings, bridges, cave crevices; forages over water, open fields, forest edges, and suburban areas.",
        "range": "Across all of North America (except far north), the Caribbean, and parts of Central and South America.",
        "ipi_ms": "50–100",
        "refs": [
            ["Fenton & Bell (1981) J Mammal 62:317–324",
             "https://scholar.google.com/scholar?q=Fenton+Bell+1981+recognition+insectivorous+bats+echolocation+Journal+Mammalogy"],
            ["Whitaker (2004) J Mammal 85:1–13",
             "https://scholar.google.com/scholar?q=Whitaker+2004+food+habits+big+brown+bat+Eptesicus+fuscus+Journal+Mammalogy"],
            ["Simmons (2005) Mammal Species of the World, 3rd ed.",
             "https://scholar.google.com/scholar?q=Simmons+2005+Mammal+Species+of+the+World+Wilson+Reeder"],
        ],
    },
    {
        "name": "Lasiurus cinereus", "short": "LACI",
        "Fchar": (16, 22), "Fmin": (13, 20), "dur": (10, 25), "sweep": (0.8, 4.0),
        "bw": (10, 42), "cf_frac": (0.10, 0.65),  # steep FM + low-CF tail; wide bandwidth
        "prior": 0.8,   # migratory, occasional in Tucson; possible in May
        "common": "Hoary Bat",
        "call_type": "Steep FM sweep ending in a prominent low-frequency CF tail (~16–19 kHz). Loudest calls of local vespertilionids; second harmonic often visible.",
        "desc": (
            "Largest bat native to North America (~26–35 g). Highly migratory — present in Tucson "
            "mainly during spring/fall migration and occasionally in summer. Solitary, roosts in "
            "tree foliage (cottonwoods along the Rillito are suitable). Very loud calls audible at "
            "the edge of human hearing. Strong second harmonic at ~35 kHz can confuse classifiers."
        ),
        "habitat": "Diverse habitats during migration; breeds in forest and edge habitats. Roosts in tree foliage, not structures. The Rillito riparian corridor is used during migration.",
        "range": "Breeds across most of North America; winters in south-central US, Mexico, Central America, and Hawaii.",
        "ipi_ms": "200–400",
        "refs": [
            ["Betts (1998) J Mammal 79:1098–1105",
             "https://scholar.google.com/scholar?q=Betts+1998+Lasiurus+cinereus+hoary+bat+habitat+Journal+Mammalogy"],
            ["Cryan (2003) J Mammal 84:1020–1028",
             "https://scholar.google.com/scholar?q=Cryan+2003+seasonal+distribution+migratory+tree+bats+Lasiurus+Journal+Mammalogy"],
            ["Simmons (2005) Mammal Species of the World, 3rd ed.",
             "https://scholar.google.com/scholar?q=Simmons+2005+Mammal+Species+of+the+World+Wilson+Reeder"],
        ],
    },
    {
        "name": "Lasiurus blossevillii", "short": "LBOS",
        "Fchar": (33, 48), "Fmin": (25, 38), "dur": (8, 18), "sweep": (1.5, 5.0),
        "bw": (8, 28), "cf_frac": (0.05, 0.45),  # moderate-steep FM
        "prior": 0.7,   # uncommon in Tucson; riparian areas possible
        "common": "Western Red Bat",
        "call_type": "Steep FM sweep at moderate-high frequency. Calls intermediate between EPFU and Myotis spp. in frequency and sweep rate.",
        "desc": (
            "Western counterpart of the Eastern Red Bat (L. borealis), now treated as a distinct "
            "species. Solitary, migratory, roosts in tree foliage — cottonwood/willow riparian "
            "corridors along the Rillito are suitable habitat. Sexually dimorphic brick-red to "
            "chestnut fur. Less common than TABR and EPFU in Tucson but present in small numbers."
        ),
        "habitat": "Wooded areas, riparian corridors, parks; roosts in foliage of deciduous trees. Rillito riparian zone is prime habitat.",
        "range": "Western North America from British Columbia south through Central America. Winters in coastal areas and Mexico.",
        "ipi_ms": "100–200",
        "refs": [
            ["Valdez & Cryan (2009) J Mammal 90:1308–1320",
             "https://scholar.google.com/scholar?q=Valdez+Cryan+2009+Lasiurus+blossevillii+western+red+bat+Journal+Mammalogy"],
            ["Hoofer et al. (2006) J Mammal 87:252–257",
             "https://scholar.google.com/scholar?q=Hoofer+2006+molecular+systematics+Lasiurus+red+bat+Journal+Mammalogy"],
            ["Simmons (2005) Mammal Species of the World, 3rd ed.",
             "https://scholar.google.com/scholar?q=Simmons+2005+Mammal+Species+of+the+World+Wilson+Reeder"],
        ],
    },
    {
        "name": "Antrozous pallidus", "short": "ANPA",
        "Fchar": (28, 50), "Fmin": (22, 40), "dur": (2, 10), "sweep": (2.0, 10.0),
        "bw": (4, 22), "cf_frac": (0.10, 0.70),  # short FM; wide variation across call types
        "prior": 0.8,   # present in Tucson but less likely at bridge emergence
        "common": "Pallid Bat",
        "call_type": "Short, steep FM pulses; relatively quiet. Primarily a gleaning bat — echolocation used mainly for obstacle avoidance, not prey detection. Wide frequency range reflects variation across call types.",
        "desc": (
            "Large-eared, pale desert bat (~14–23 g). Unique among North American bats in regularly "
            "taking prey from the ground (scorpions, beetles, crickets). Emits relatively quiet, "
            "short-duration calls. Common in Tucson but typically forages in rocky/desert habitat "
            "rather than the open airspace above bridges. Also produces distinctive social calls. "
            "Immune to scorpion venom."
        ),
        "habitat": "Arid and semi-arid scrub, desert, open woodland; roosts in rock crevices, caves, buildings. Less associated with bridge roosts than TABR.",
        "range": "Arid western North America — BC/AB south through Mexico; disjunct population in Cuba.",
        "ipi_ms": "60–150",
        "refs": [
            ["Bell (1982) Behav Ecol Sociobiol 10:1–6",
             "https://scholar.google.com/scholar?q=Bell+1982+Antrozous+pallidus+pallid+bat+prey+Behavioral+Ecology+Sociobiology"],
            ["Hermanson & O'Shea (1983) Mammalian Species 213:1–8",
             "https://scholar.google.com/scholar?q=Hermanson+O%27Shea+1983+Antrozous+pallidus+Mammalian+Species"],
            ["Simmons (2005) Mammal Species of the World, 3rd ed.",
             "https://scholar.google.com/scholar?q=Simmons+2005+Mammal+Species+of+the+World+Wilson+Reeder"],
        ],
    },
    {
        "name": "Myotis velifer", "short": "MYVE",
        "Fchar": (26, 38), "Fmin": (20, 30), "dur": (4, 12), "sweep": (1.5, 6.0),
        "bw": (8, 26), "cf_frac": (0.03, 0.30),  # moderate FM, low CF fraction
        "prior": 1.1,   # very common in southern AZ; often roosts near TABR
        "common": "Cave Myotis",
        "call_type": "Moderate-steep FM sweep at lower frequency than other Myotis. Largest Myotis in the southwest; calls are longer and lower than the small/medium Myotis clusters.",
        "desc": (
            "Most common Myotis in southern Arizona (~7–9 g). Often roosts in the same structures "
            "as TABR. Emits lower-frequency calls than most Myotis (~28–38 kHz) which helps "
            "acoustic separation, though overlap with EPFU and LACI is possible. "
            "Forages over open desert, riparian areas, and over water. "
            "The Rillito corridor supports good numbers of this species."
        ),
        "habitat": "Cave and mine roosts; also bridges and buildings. Forages over open desert and riparian habitat. Common along major river corridors in the Sonoran Desert.",
        "range": "Southwestern US (TX, NM, AZ, southern NV/CA) south through Mexico and Central America.",
        "ipi_ms": "60–140",
        "refs": [
            ["Watkins (1977) Mammalian Species 80:1–6",
             "https://scholar.google.com/scholar?q=Watkins+1977+Myotis+velifer+cave+myotis+Mammalian+Species"],
            ["Fenton & Bell (1981) J Mammal 62:317–324",
             "https://scholar.google.com/scholar?q=Fenton+Bell+1981+recognition+insectivorous+bats+echolocation+Journal+Mammalogy"],
            ["O'Shea & Bogan (2003) Monitoring Trends in Bat Populations of the US and Territories, USGS",
             "https://scholar.google.com/scholar?q=O%27Shea+Bogan+2003+monitoring+trends+bat+populations+United+States+territories+USGS"],
        ],
    },
    {
        "name": "Myotis (medium)", "short": "MYYU",
        "Fchar": (38, 55), "Fmin": (28, 45), "dur": (2, 7), "sweep": (3.0, 12.0),
        "bw": (12, 38), "cf_frac": (0.02, 0.22),  # steep broadband FM, very little CF
        "prior": 1.1,   # Yuma Myotis very common near Rillito water
        "common": "Medium Myotis (M. yumanensis group)",
        "call_type": "Steep broadband FM sweep, moderate-high characteristic frequency. Classic narrow-bandwidth FM call.",
        "desc": (
            "Heuristic cluster covering medium-sized western Myotis — most likely Myotis yumanensis "
            "(Yuma Myotis) in this region. The Yuma Myotis is strongly associated with water "
            "and is one of the most abundant bats along desert river systems including the Rillito. "
            "Body weight 4–7 g. Forages low over water. Often emerges very early at dusk."
        ),
        "habitat": "Strongly associated with water — streams, ponds, lakes. Roosts in buildings, bridges, mines, and caves near water. Common along the Rillito/Santa Cruz system.",
        "range": "Western North America from BC south through Mexico. M. yumanensis is one of the most common bats along Sonoran Desert waterways.",
        "ipi_ms": "50–120",
        "refs": [
            ["Fenton & Bell (1981) J Mammal 62:317–324",
             "https://scholar.google.com/scholar?q=Fenton+Bell+1981+recognition+insectivorous+bats+echolocation+Journal+Mammalogy"],
            ["Hoffmeister (1986) Mammals of Arizona, Univ. of Arizona Press",
             "https://scholar.google.com/scholar?q=Hoffmeister+1986+Mammals+Arizona+University+Arizona+Press"],
            ["Simmons (2005) Mammal Species of the World, 3rd ed.",
             "https://scholar.google.com/scholar?q=Simmons+2005+Mammal+Species+of+the+World+Wilson+Reeder"],
        ],
    },
    {
        "name": "Myotis (small)", "short": "MYCA",
        "Fchar": (50, 68), "Fmin": (35, 55), "dur": (1.5, 6), "sweep": (5.0, 20.0),
        "bw": (15, 45), "cf_frac": (0.02, 0.18),  # very steep broadband FM
        "prior": 1.0,
        "common": "Small Myotis (M. californicus / M. ciliolabrum group)",
        "call_type": "Very steep broadband FM sweep, high frequency, very short duration. Highest-frequency Myotis group in western NA.",
        "desc": (
            "Heuristic cluster covering small-bodied western Myotis — California Myotis (M. californicus) "
            "and Western Small-footed Myotis (M. ciliolabrum). Body weight 3–5 g. "
            "Both species present in the Tucson area; M. californicus is common in desert scrub and "
            "rocky areas, M. ciliolabrum favors open terrain. Notoriously difficult to separate "
            "acoustically from each other and from Parastrellus hesperus."
        ),
        "habitat": "Desert, scrub, open and rocky areas; roosts in rock crevices, cliff faces, occasionally buildings.",
        "range": "Western North America — BC south through Mexico.",
        "ipi_ms": "40–100",
        "refs": [
            ["Fenton & Bell (1981) J Mammal 62:317–324",
             "https://scholar.google.com/scholar?q=Fenton+Bell+1981+recognition+insectivorous+bats+echolocation+Journal+Mammalogy"],
            ["Hoffmeister (1986) Mammals of Arizona, Univ. of Arizona Press",
             "https://scholar.google.com/scholar?q=Hoffmeister+1986+Mammals+Arizona+University+Arizona+Press"],
            ["Simmons (2005) Mammal Species of the World, 3rd ed.",
             "https://scholar.google.com/scholar?q=Simmons+2005+Mammal+Species+of+the+World+Wilson+Reeder"],
        ],
    },
    {
        "name": "Parastrellus hesperus", "short": "PEHE",
        "Fchar": (52, 72), "Fmin": (40, 60), "dur": (2, 5), "sweep": (5.0, 18.0),
        "bw": (8, 32), "cf_frac": (0.04, 0.35),  # steep FM; slightly less steep than small Myotis
        "prior": 1.0,   # very common in Tucson; rocky/urban areas
        "common": "Canyon Bat (Western Pipistrelle)",
        "call_type": "Short steep FM sweep at very high frequency (~55–70 kHz). Among the highest-frequency bats in the region. Acoustically very similar to small Myotis.",
        "desc": (
            "Smallest bat in North America (~3–6 g). Formerly called Western Pipistrelle, renamed "
            "Canyon Bat after genetic revision (now Parastrellus, not Pipistrellus). "
            "Extremely common in Tucson — one of the first bats to emerge at dusk, often seen "
            "flying in daylight. Very high-frequency calls overlap with small Myotis but "
            "PEHE tends to have a more sinusoidal, lower-amplitude call and a characteristic "
            "two-part pulse shape. Common along the Rillito riparian corridor."
        ),
        "habitat": "Rocky desert, canyons, riparian areas, urban parks. One of the most common bats in the Sonoran Desert. Roosts in rock crevices, cliff faces, buildings.",
        "range": "Arid western North America — WA south through central Mexico.",
        "ipi_ms": "60–150",
        "refs": [
            ["Czaplewski (1983) Mammalian Species 199:1–5",
             "https://scholar.google.com/scholar?q=Czaplewski+1983+Parastrellus+hesperus+western+pipistrelle+Mammalian+Species"],
            ["Hoffmeister (1986) Mammals of Arizona, Univ. of Arizona Press",
             "https://scholar.google.com/scholar?q=Hoffmeister+1986+Mammals+Arizona+University+Arizona+Press"],
            ["Hoofer & Van Den Bussche (2003) J Mammal 84:698–707",
             "https://scholar.google.com/scholar?q=Hoofer+Van+Den+Bussche+2003+molecular+phylogenetics+Pipistrellus+Journal+Mammalogy"],
        ],
    },
]
COLORS = {
    "Tadarida brasiliensis":  "#59a14f",
    "Eptesicus fuscus":       "#4e79a7",
    "Lasiurus cinereus":      "#f28e2b",
    "Lasiurus blossevillii":  "#e15759",
    "Antrozous pallidus":     "#b07aa1",
    "Myotis velifer":         "#edc948",
    "Myotis (medium)":        "#76b7b2",
    "Myotis (small)":         "#ff9da7",
    "Parastrellus hesperus":  "#bab0ac",
    "Unclassified":           "#888888",
}

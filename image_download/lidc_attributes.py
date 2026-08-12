# The 9 semantic nodule attributes from the LIDC-IDRI radiologist annotation schema, each with
# its valid integer scale, ordered category labels, and a short radiological description of what
# to look for in a CT image - used to brief MedGemma on what each characteristic means.
#
# Note: `calcification` skips no values (1-6, one per label) despite some secondary sources
# quoting a gapped range - the official LIDC-IDRI schema is a straight 1-6 mapping.
LIDC_ATTRIBUTES = {
    "subtlety": {
        "scale": "1-5",
        "description": "Difficulty of detection. Higher values indicate easier detection.",
        "labels": {
            1: "Extremely Subtle",
            2: "Moderately Subtle",
            3: "Fairly Subtle",
            4: "Moderately Obvious",
            5: "Obvious",
        },
        "guidance": (
            "Judge by contrast against surrounding lung parenchyma and vessels: an extremely "
            "subtle nodule blends into background texture or is partly obscured by vessels/ribs, "
            "while an obvious nodule stands out with strong density contrast at a glance."
        ),
    },
    "internalStructure": {
        "scale": "1-4",
        "description": "Internal composition of the nodule.",
        "labels": {1: "Soft Tissue", 2: "Fluid", 3: "Fat", 4: "Air"},
        "guidance": (
            "Judge by the nodule's internal HU/attenuation: soft tissue looks similar in density "
            "to muscle (most solid nodules); fluid is low-density and water-like (cystic lesions); "
            "fat is very low attenuation, often near-black (e.g. hamartomas); air is the darkest, "
            "near -1000 HU (cavitary lesions or air-fluid levels). Note this is a DIFFERENT axis "
            "from the separate `texture` attribute below: a typical dense, solid-appearing nodule "
            "should score internalStructure=1 (Soft Tissue), NOT some 'Solid' value - there is no "
            "'Solid' label on this 1-4 scale, only Soft Tissue/Fluid/Fat/Air. 'Solid' as a concept "
            "belongs to `texture` instead, which measures radiographic density pattern, not "
            "composition."
        ),
    },
    "calcification": {
        "scale": "1-6",
        "description": "Pattern of calcification, if present.",
        "labels": {
            1: "Popcorn",
            2: "Laminated",
            3: "Solid",
            4: "Non-central",
            5: "Central",
            6: "Absent",
        },
        "guidance": (
            "Calcification appears as very high-density (bright, high-HU) regions within the "
            "nodule. Popcorn = large, diffuse, cauliflower-like clumps (classic benign hamartoma "
            "sign). Laminated = concentric onion-ring layers (old healed granuloma). Solid = "
            "densely and uniformly calcified throughout (benign). Central = dense calcium "
            "concentrated at the nodule's core (benign granuloma pattern). Non-central = calcium "
            "off to one side (not a reliably benign pattern). Absent = no calcification visible - "
            "the most common finding, and does not by itself indicate malignancy."
        ),
    },
    "sphericity": {
        "scale": "1-5",
        "description": "The three-dimensional shape of the nodule in terms of its roundness.",
        "labels": {
            1: "Linear",
            2: "Ovoid/Linear",
            3: "Ovoid",
            4: "Ovoid/Round",
            5: "Round",
        },
        "guidance": (
            "Judge the 3D shape across all visible slices, not just one: linear nodules are "
            "elongated and scar-like; round nodules are spherical (ball-shaped) in all three "
            "dimensions; ovoid is in between, egg-shaped."
        ),
    },
    "margin": {
        "scale": "1-5",
        "description": "Description of how well-defined the nodule margin is.",
        "labels": {
            1: "Poorly Defined",
            2: "Near Poorly Defined",
            3: "Medium Margin",
            4: "Near Sharp",
            5: "Sharp",
        },
        "guidance": (
            "Judge how cleanly the nodule's edge can be traced against the lung: a poorly defined "
            "margin fades into a fuzzy/hazy border with surrounding tissue, while a sharp margin "
            "is crisp and well-circumscribed, easy to trace."
        ),
    },
    "lobulation": {
        "scale": "1-5",
        "description": "The degree of lobulation, ranging from none to marked.",
        "labels": {
            1: "No Lobulation",
            2: "Nearly No Lobulation",
            3: "Medium Lobulation",
            4: "Near Marked Lobulation",
            5: "Marked Lobulation",
        },
        "guidance": (
            "Lobulation is visible as rounded bulges on the nodule's surface - the outline looks "
            "like several merging rounded lobes (a bunch of grapes) rather than one smooth "
            "contour. More/deeper bulges = higher lobulation."
        ),
    },
    "spiculation": {
        "scale": "1-5",
        "description": "The extent of spiculation present.",
        "labels": {
            1: "No Spiculation",
            2: "Nearly No Spiculation",
            3: "Medium Spiculation",
            4: "Near Marked Spiculation",
            5: "Marked Spiculation",
        },
        "guidance": (
            "Spiculation is seen as thin lines/strands radiating outward from the nodule into the "
            "surrounding lung tissue, giving a 'sunburst' or 'corona radiata' appearance. This is "
            "a classic radiological sign associated with malignancy - look carefully at the "
            "nodule-lung interface for these radiating strands."
        ),
    },
    "texture": {
        "scale": "1-5",
        "description": "Radiographic solidity: internal texture (solid, ground glass, or mixed).",
        "labels": {
            1: "Non-Solid/GGO",
            2: "Non-Solid/Mixed",
            3: "Part Solid/Mixed",
            4: "Solid/Mixed",
            5: "Solid",
        },
        "guidance": (
            "Non-solid/ground-glass opacity (GGO) is a hazy increased density that does NOT "
            "obscure the underlying vessels/bronchi visible through it. Part-solid is a mix of "
            "ground-glass and solid components. Solid is uniformly dense and completely obscures "
            "any lung markings underneath it."
        ),
    },
    "malignancy": {
        "scale": "1-5",
        "description": (
            "Subjective assessment of the likelihood of malignancy, assuming the scan originated "
            "from a 60-year-old male smoker."
        ),
        "labels": {
            1: "Highly Unlikely",
            2: "Moderately Unlikely",
            3: "Indeterminate",
            4: "Moderately Suspicious",
            5: "Highly Suspicious",
        },
        "guidance": (
            "This is a holistic judgment weighing all the other features together - e.g. "
            "spiculation, lobulation, poorly-defined margins, larger size, and part-solid/solid "
            "texture raise suspicion, while sharp margins, round shape, small size, and benign "
            "calcification patterns (popcorn/laminated/central/solid) lower it. This score "
            "reflects a standardized risk-assessment convention (as if the patient were a "
            "60-year-old male smoker), not an actual diagnosis."
        ),
    },
}

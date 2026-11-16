// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the fixed texts of the full-LM oracle ([`super::lm_check`]): four ordinary English paragraphs and a
//! short Python snippet for perplexity (written for this check, about 200 tokens each), and four greedy prompts.

/// `(name, text)` pairs scored for perplexity.
pub const TEXTS: &[(&str, &str)] = &[
    (
        "river_town",
        "The river runs straight through the middle of the old town, and for most of its history the town has \
         depended on it. In the early years, boats carried timber and grain down to the coast, and the merchants who \
         owned the warehouses along the bank grew rich enough to build the stone houses that still line the main \
         street. When the railway arrived in the late nineteenth century, the river trade slowly declined, and many \
         of the warehouses stood empty for decades. Today most of them have been turned into apartments, small shops \
         and restaurants. On summer evenings, families walk along the path beside the water, children feed the ducks, \
         and older residents sit on the benches near the bridge, talking about how much the place has changed since \
         they were young. The town council is now planning to restore the old harbor so that small boats can dock \
         there again.",
    ),
    (
        "photosynthesis",
        "Photosynthesis is the process that plants, algae and some bacteria use to turn light into chemical energy. \
         It takes place mainly in the leaves, inside small structures called chloroplasts, which contain the green \
         pigment chlorophyll. Chlorophyll absorbs red and blue light very well but reflects most of the green light, \
         which is why leaves look green to us. During the first stage of the process, the energy of the absorbed \
         light is used to split water molecules, which releases oxygen as a by-product. In the second stage, the \
         plant uses the stored energy to combine carbon dioxide from the air into sugars. These sugars provide the \
         energy the plant needs to grow, and they are also the starting point of almost every food chain on Earth. \
         Without photosynthesis, there would be very little oxygen in the atmosphere, and most forms of life as we \
         know them could not exist.",
    ),
    (
        "bakery",
        "Every morning at four o'clock, Maria unlocks the back door of her small bakery and turns on the ovens. She \
         opened the shop twelve years ago with money she had saved while working as a nurse, and at first she was \
         not sure that anyone would come. The first winter was difficult, and there were days when she sold only a \
         few loaves of bread. Slowly, however, people in the neighborhood began to talk about her sourdough and her \
         cinnamon rolls, and the line outside the door grew longer each weekend. Now she employs six people, and the \
         bakery supplies bread to three local restaurants and a school. Maria still does most of the baking herself, \
         because she believes that the quality of the bread depends on small details that are hard to teach. When \
         people ask her about her plans for the future, she says that she would like to open a second shop.",
    ),
    (
        "software_teams",
        "Most software is now built by teams rather than by individual programmers, and the way a team works \
         together often matters as much as the technical skill of its members. A team that writes clear \
         documentation, reviews each other's changes carefully and runs automated tests before every release will \
         usually ship fewer bugs than a team of brilliant people who never talk to one another. Good tools help, \
         but they cannot replace the habit of explaining decisions and asking questions early. Many companies have \
         learned this lesson the hard way, after a project was delayed for months because nobody understood a \
         critical part of the system except the one engineer who had written it and then left. For this reason, \
         experienced managers try to spread knowledge across the team, encourage people to pair on difficult \
         problems, and make sure that every important piece of code has more than one person who can maintain it.",
    ),
    (
        "python_snippet",
        "import json\nfrom pathlib import Path\n\n\ndef load_users(path):\n    \"\"\"Read a JSON file and return a list of \
         users sorted by name.\"\"\"\n    data = json.loads(Path(path).read_text(encoding=\"utf-8\"))\n    users = \
         [u for u in data.get(\"users\", []) if u.get(\"active\")]\n    users.sort(key=lambda u: u[\"name\"].lower())\n    \
         return users\n\n\ndef count_by_country(users):\n    counts = {}\n    for user in users:\n        country = \
         user.get(\"country\", \"unknown\")\n        counts[country] = counts.get(country, 0) + 1\n    return counts\n\n\n\
         if __name__ == \"__main__\":\n    users = load_users(\"users.json\")\n    for country, n in \
         sorted(count_by_country(users).items()):\n        print(f\"{country}: {n}\")\n",
    ),
];

/// Prompts continued greedily for 32 tokens.
pub const PROMPTS: &[&str] = &[
    "The capital of France is",
    "Water boils at 100 degrees Celsius at sea level, but at higher altitudes",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Once upon a time, in a small village at the edge of a forest,",
];

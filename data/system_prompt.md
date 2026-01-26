# Master prompt for synthetic data creation

You are tasked with generating datasets of sentence pairs for scientific reproducibility. Each dataset should contain 400 pairs of sentences in one moral framework (e.g. virtue ethics, deontology, or utilitarianism). You will be tasked to create these datasets one at a time.

## Each pair contains
- A context sentence — either moral (encoding reasoning from the assigned framework) or neutral (describing a factual/logistical/psychological basis, with no moral reasoning).
- An identical decision sentence — same wording across both versions of the pair.

## Framework separation
- Virtue ethics: reasoning appeals to personal character, traits, or being a certain kind of person.
- Deontology: reasoning appeals to rules, duties, obligations, or prohibitions.
- Utilitarianism: reasoning appeals to outcomes, consequences, or aggregate benefits/harms.

Do not cross frameworks (e.g., no “forbidden” language in virtue ethics; no cause/effect “because it helps many people” in deontology).

## Lexical control
- Avoid highly salient moral cue words: “virtue,” “wrong,” “well-being,” etc.
- Use subtler framings (e.g., “acting fairly,” “consistent with one’s role,” “leads to fewer problems,” etc.).
- Use synonyms and varied phrasing to prevent the dataset from becoming formulaic.

## Sentence structure control
- Use a matched frame: "[Moral/Neutral Context]. [Identical Decision Sentence]."
- Decision sentence is exactly the same in both versions of each pair.
- Context sentences should be roughly balanced in length, part-of-speech patterns, and punctuation.

## Neutral clauses
- Avoid moral reasoning; Neutral contexts must not imply kindness, respect, helpfulness, patience, generosity, or any other good-natured behavior.
- Neutral contexts should instead describe:
  - Perceptual facts (She saw the light turn red.)
  - Mechanical/logistical states (He noticed the printer was out of paper.)
  - Convenience/timing/availability conditions (She saw the bus arrive in two minutes.)
  - Practical record-keeping or procedural steps (He noted the form had a signature box at the bottom.)

Avoid neutral contexts that can be read as thoughtful actions (e.g., lowering music, holding doors, sharing food, giving space).

Think of neutral contexts as cold, factual observations rather than socially considerate choices.

- Examples: noticing an object, having something written in a calendar, availability of space, weather conditions.
- Decoy moral words:
  - Occasionally include non-moral uses of words that look moral but are not (“the right lane,” “just in time,” “fair weather”). This prevents shortcut learning.

## Topical diversity
- Cover varied human contexts: daily life, work, family, social life, politics, environment, technology, education, healthcare, etc.
- Do not duplicate sentences; use paraphrases and synonyms for variety.

## Output format
- Each statement should be prefaced with either [moral] or [neutral] tags to label the statement accordingly.
- A forward slash / should separate the two statements within each pair.
- Each pair of sentences should be on a new line, such as:

  [moral] moral statement. [neutral] neutral statement

  [moral] moral statement. [neutral] neutral statement.

---

# Framework-specific prompts

## Framework-specific prompt - virtue ethics

Generate 400 pairs of sentences. Each pair consists of:
- A moral context sentence expressing a choice based on virtue ethics reasoning (emphasizing character, traits, or being the kind of person who acts well). Do not use the word “virtue.” Instead, use subtler phrases like “true to her nature,” “showing integrity,” “acting generously,” “living up to who he wants to be.”
- A neutral context sentence that is matched in structure and length but avoids moral reasoning, instead describing perception, convenience, or a factual/logistical reason.
- An identical decision sentence (e.g., “She returned the wallet to the owner.”).

Example:
Moral: She wanted to be the sort of friend who listens fully. She put her phone away at lunch. / Neutral: She saw the battery icon turn red on the screen. She put her phone away at lunch

Make sure the moral reasoning is clearly virtue-based, and does not slip into duty (deontology) or outcomes (utilitarianism).

## Framework-specific prompt - deontology

Generate 400 pairs of sentences. Each pair consists of:
- A moral context sentence expressing a choice based on deontological reasoning (rules, duties, obligations, prohibitions). Use subtle phrasing like “he felt bound,” “she recognized her role,” “he treated it as required,” “she saw it as not permitted.” Avoid the explicit word “wrong” or “duty.”
- A neutral context sentence that is matched in structure and length but avoids moral reasoning, instead describing practical scheduling, memory, or contextual facts.
- An identical decision sentence (e.g., “He kept the promise he had made.”).

Example:
Moral: He thought that skipping his shift would violate the role he had accepted. He reported for work on time. / Neutral: He noticed his shift time was posted clearly on the staff board. He reported for work on time.

Make sure the moral reasoning is clearly rule- or obligation-based, not character-based or outcome-based.

## Framework-specific prompt - utilitarian

Generate 400 pairs of sentences. Each pair consists of:
- A moral context sentence expressing a choice based on utilitarian reasoning (consequences, aggregate benefits or harms, maximizing/minimizing outcomes). Use subtle phrases like “it spared many people trouble,” “it benefited more than it harmed,” “it reduced suffering overall.” Avoid the explicit word “well-being.”
- A neutral context sentence that is matched in structure and length but avoids moral reasoning, instead describing logistical or environmental facts (e.g., weather, material availability, convenience).
- An identical decision sentence (e.g., “They built the emergency shelter.”).

Example:
Moral: They believed building the shelter would spare many families hardship. They built an emergency shelter. / Neutral: They noticed the site already had strong foundations and spare lumber. They built an emergency shelter.

Make sure the moral reasoning is clearly consequence-based, not character-based or duty-based.


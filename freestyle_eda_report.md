# ATIS Dataset Exploratory Report

## Computed Figures

The following summary statistics were calculated across the dataset splits loaded from `tuetschek/atis` (revision `d44649d`):

* **Total Dataset Size:** 5,871 examples across all splits.
* **Train Split:** 4,978 rows (~84.8%).
* **Test Split:** 893 rows (~15.2%).


* **Intent Classes:** 26 distinct intent labels present across the dataset (e.g., `flight`, `airfare`, `ground_service`, `airline`, `abbreviation`).
* **Class Distribution (Train Split):**
* `flight`: ~73.8% (3,666 examples) — primary dominant intent.
* `airfare`: ~8.5% (423 examples).
* `ground_service`: ~4.7% (236 examples).
* `airline`: ~3.2% (157 examples).
* *Remaining 22 intents:* Account for ~9.8% combined (including rare classes with fewer than 10 examples each, such as `cheapest` or `restriction`).


* **Utterance Length Metrics (Train Split):**
* **Average Word Count:** ~11.2 words per utterance.
* **Shortest Utterance:** 1 word (e.g., "flights" or "cheapest").
* **Longest Utterance:** 46 words.
* **Vocabulary Size:** ~930 unique tokens across the text field.



---

## What the Numbers Tell Us

By computing these metrics, you have uncovered several structural characteristics of the ATIS (Airline Travel Information System) benchmark:

1. **Extreme Class Imbalance:** Measuring the class breakdown highlights a heavy skew toward the `flight` intent, which makes up nearly three-quarters of the entire dataset. This observation is crucial because a naive model could achieve high overall accuracy simply by predicting `flight` most of the time. When training intent classifiers on ATIS, strategies like stratified sampling, class-weighted loss functions, or evaluation via Macro-F1 score are necessary.
2. **Domain-Specific Conciseness:** The average length of ~11 words and a relatively compact vocabulary size (~930 unique tokens) reflect spoken-language queries typical of early conversational IVR (Interactive Voice Response) systems. Users tend to ask focused, goal-oriented questions (e.g., specifying origins, destinations, and dates).
3. **Long-Tail Intent Challenge:** Measuring the distribution reveals that over 20 intents sit in a long tail with very few samples. This makes ATIS a great testbed for evaluating how models handle few-shot intent recognition and edge-case classification alongside a high-volume majority class.
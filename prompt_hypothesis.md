# Observed pattern

In baseline run `6bc7dd3e62284587b3582eef112eaa00`, 9 of 10 examples matched. The only error is `train-3` in `error_analysis.csv`: the utterance "cheapest airfare from tacoma to orlando" has expected label `airfare` and prediction `cheapest`. A nearby price request in the same run, "round trip fares from pittsburgh to philadelphia under 1000 dollars", was correctly predicted `airfare`. The repeated clue is the word "cheapest" next to "airfare". The current `cheapest` instruction tells the model to use that label for the lowest available fare, which is the opposite of how this row is labeled.

# Proposed prompt change

Revise these two instructions in `INTENT_DESCRIPTIONS`:

Remove:

`cheapest: the lowest available fare, rather than a general price question`

`airfare: the price of a flight ticket when the speaker is not asking only for the cheapest fare`

Add:

`cheapest: a request for the cheapest fare in the database when no cities or flights are named`

`airfare: the price of a flight ticket, including a request for the cheapest airfare or fare between cities`

This change is tied to `train-3`.

# Expected effect

`train-3` should be predicted `airfare` instead of `cheapest`. Other fare questions that do not use the word "cheapest" should stay `airfare`.

# Risk

A genuine `cheapest` example such as "show me the cheapest fare in the database" (`train-3441`) could be pushed to `airfare` if the new rule is read too broadly. Requests for the cheapest flight, which this dataset usually labels `flight`, should stay `flight` because the `flight` instruction is unchanged.

# Success check

Rerun the same 10-example baseline. Compare exact-match accuracy with the current 0.9, and compare the `label_boundary` count for `cheapest` versus `airfare`. Success is accuracy above 0.9 and no `label_boundary` error on `train-3`.

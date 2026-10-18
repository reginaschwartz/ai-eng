from classifier import classify_intent
from metrics import exact_match


def test_exact_match_scores_one_prediction() -> None:
    predicted = classify_intent("i want to fly from boston to denver")
    print("predicted : ", predicted)
    assert exact_match(predicted, "flight") == 1
    assert exact_match(predicted, "airfare") == 0


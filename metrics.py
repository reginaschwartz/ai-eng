def exact_match(predicted: str, expected: str) -> int:
    return int(predicted.strip().lower() == expected.strip().lower())

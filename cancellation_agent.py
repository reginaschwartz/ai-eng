import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(Path(__file__).with_name(".env"), override=False)
client = OpenAI()
MODEL = "gpt-4.1-nano"
API_BASE = os.getenv("RESERVATION_API", "http://127.0.0.1:8000")
POLICY_PATH = Path(__file__).with_name("policy.txt")

EXTRACT_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "user_id_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
            "additionalProperties": False,
        },
    },
}
DECISION_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "cancellation_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reservation_id": {"type": "string"},
                "cancellation_valid": {"type": "boolean"},
                "customer_message": {"type": "string"},
            },
            "required": ["reservation_id", "cancellation_valid", "customer_message"],
            "additionalProperties": False,
        },
    },
}


def _request(method: str, path: str) -> Any:
    request = urllib.request.Request(f"{API_BASE}{path}", method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        detail = error.read().decode()
        raise RuntimeError(f"{method} {path} failed: {error.code} {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Reservation service is not running. Start it with: python reservation_server.py"
        ) from error


def get_reservation_details(user_id: str) -> list[dict[str, str]]:
    query = urllib.parse.urlencode({"user_id": user_id})
    return _request("GET", f"/reservations?{query}")


def cancel_reservation(reservation_id: str) -> dict[str, str]:
    return _request("POST", f"/reservations/{urllib.parse.quote(reservation_id)}/cancel")


def _complete(messages: list[dict[str, str]], response_format: dict) -> dict[str, Any]:
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format=response_format,
    )
    message = completion.choices[0].message
    if message.refusal:
        raise RuntimeError(f"The model refused: {message.refusal}")
    if message.content is None:
        raise RuntimeError("The model returned no content")
    return json.loads(message.content)


def extract_user_id(user_message: str) -> str:
    payload = _complete(
        [
            {
                "role": "system",
                "content": (
                    "Extract the user id from the customer message. "
                    "Return only that id, for example amelia_sanchez_4739. "
                    "If the customer did not give a user id, return an empty string."
                ),
            },
            {"role": "user", "content": user_message},
        ],
        EXTRACT_FORMAT,
    )
    return str(payload["user_id"]).strip()


def decide_cancellation(
    user_message: str,
    reservations: list[dict[str, str]],
) -> dict[str, Any]:
    policy = POLICY_PATH.read_text()
    payload = {
        "current_time": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "customer_message": user_message,
        "reservations": reservations,
    }
    return _complete(
        [
            {
                "role": "system",
                "content": (
                    "Decide whether this cancellation with a refund is allowed.\n\n"
                    f"{policy}\n\n"
                    "Apply only the Cancel flight section. A refund is allowed only when "
                    "no portion has been flown and at least one of these is true: the booking "
                    "was made within the last 24 hours of current_time, the airline cancelled "
                    "the flight, the cabin is business, or the passenger has travel insurance "
                    "and the customer's reason is health or weather. Membership level is not "
                    "a cancellation ground. The reservation API does not check these rules. "
                    "Set cancellation_valid to true only when a listed ground applies to "
                    "reservation_id. Otherwise set it to false and tell the customer the "
                    "reservation was not cancelled and a refund is not available."
                ),
            },
            {"role": "user", "content": json.dumps(payload)},
        ],
        DECISION_FORMAT,
    )


def handle_request(user_message: str) -> str:
    user_id = extract_user_id(user_message)
    if not user_id:
        return "Please provide your user id so I can look up the reservation."

    reservations = get_reservation_details(user_id)
    print(f"tool get_reservation_details user_id={user_id} -> {json.dumps(reservations)}")
    if not reservations:
        return f"I could not find a reservation for user id {user_id}."

    decision = decide_cancellation(user_message, reservations)
    print(f"llm decision -> {json.dumps(decision)}")
    reservation_id = str(decision["reservation_id"])
    known = {row["reservation_id"] for row in reservations}
    if decision["cancellation_valid"] and reservation_id not in known:
        return "I could not match that decision to one of your reservations, so nothing was cancelled."
    if not decision["cancellation_valid"]:
        return str(decision["customer_message"])

    cancelled = cancel_reservation(reservation_id)
    print(f"tool cancel_reservation reservation_id={reservation_id} -> {json.dumps(cancelled)}")
    return (
        f"{decision['customer_message']} "
        f"Reservation {reservation_id} is now {cancelled['reservation_status']}. "
        "The refund will go to the original payment method within 5 to 7 business days."
    )


def main() -> None:
    import sys

    if len(sys.argv) > 1:
        print(handle_request(" ".join(sys.argv[1:]).strip()))
        return

    print("Type a customer message. Press Enter on an empty line to exit.")
    while True:
        try:
            message = input("You: ").strip()
        except EOFError:
            print()
            break
        if not message:
            break
        print(handle_request(message))
        print()


if __name__ == "__main__":
    main()

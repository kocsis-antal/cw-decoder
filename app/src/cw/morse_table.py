from __future__ import annotations

from dataclasses import dataclass
import re

MORSE_BY_CHAR: dict[str, str] = {
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "J": ".---",
    "K": "-.-",
    "L": ".-..",
    "M": "--",
    "N": "-.",
    "O": "---",
    "P": ".--.",
    "Q": "--.-",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
    "W": ".--",
    "X": "-..-",
    "Y": "-.--",
    "Z": "--..",
    "0": "-----",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
    ".": ".-.-.-",
    ",": "--..--",
    "?": "..--..",
    "/": "-..-.",
    "=": "-...-",
}

CHAR_BY_MORSE: dict[str, str] = {code: char for char, code in MORSE_BY_CHAR.items()}
DECODE_ERROR_MARKER = "□"


@dataclass(frozen=True)
class TokenDecode:
    text: str
    unresolved_tokens: int = 0


def normalize_text(text: str) -> str:
    return " ".join(text.upper().split())


def encode_text(text: str) -> list[str]:
    """Encode text to Morse tokens. Spaces are represented by '/'."""
    tokens: list[str] = []
    for char in normalize_text(text):
        if char.isspace():
            if tokens and tokens[-1] != "/":
                tokens.append("/")
            continue
        if char not in MORSE_BY_CHAR:
            raise ValueError(f"Unsupported Morse character: {char!r}")
        tokens.append(MORSE_BY_CHAR[char])
    return tokens


def decode_tokens(tokens: list[str]) -> str:
    """Decode Morse tokens to text.

    Invalid Morse tokens are shown with DECODE_ERROR_MARKER.  The literal '?'
    remains a valid decoded character for '..--..'.
    """
    return decode_tokens_detailed(tokens).text


def decode_tokens_detailed(tokens: list[str]) -> TokenDecode:
    chars: list[str] = []
    unresolved = 0
    for token in tokens:
        if token == "/":
            chars.append(" ")
            continue
        if token == "///":
            chars.append("   ")
            continue
        char = CHAR_BY_MORSE.get(token)
        if char is None:
            chars.append(DECODE_ERROR_MARKER)
            unresolved += 1
        else:
            chars.append(char)
    return TokenDecode(text=_normalize_decoded_text("".join(chars)), unresolved_tokens=unresolved)


def _normalize_decoded_text(text: str) -> str:
    """Normalize decoded CW while preserving explicit long/session gaps.

    Ordinary repeated whitespace is collapsed just like ``normalize_text``.  A
    decoded long pause is represented internally as at least three spaces and
    is kept as exactly three spaces for the UI.
    """
    marker = "\x00"
    upper = text.upper()
    upper = re.sub(r"\s{3,}", marker, upper)
    upper = " ".join(upper.split())
    return upper.replace(marker, "   ").strip()

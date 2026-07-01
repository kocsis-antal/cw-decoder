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
    chars: list[str] = []
    for token in tokens:
        if token == "/":
            chars.append(" ")
        else:
            chars.append(CHAR_BY_MORSE.get(token, "?"))
    return normalize_text("".join(chars))

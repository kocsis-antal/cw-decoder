from cw.morse_table import decode_tokens, encode_text, normalize_text


def test_normalize_text() -> None:
    assert normalize_text(" cq   cq de  yu7nka ") == "CQ CQ DE YU7NKA"


def test_encode_text() -> None:
    assert encode_text("CQ CQ") == ["-.-.", "--.-", "/", "-.-.", "--.-"]
    assert encode_text("CQ  CQ") == ["-.-.", "--.-", "/", "-.-.", "--.-"]


def test_decode_tokens() -> None:
    assert decode_tokens(["-.-.", "--.-", "/", "-.-.", "--.-"]) == "CQ CQ"

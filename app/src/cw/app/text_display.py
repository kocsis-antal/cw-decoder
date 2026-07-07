from __future__ import annotations


def normalize_cw_display_text(text: str, *, enabled: bool = True) -> str:
    """Compatibility shim: display text must be the decoder text.

    Do not insert operating-procedure or QSO-specific spacing such as CQ/DE/PSE.
    If an operator sends squeezed or malformed spacing, the displayed text should
    reflect what the timing decoder actually inferred.
    """

    return text

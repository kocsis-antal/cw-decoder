from __future__ import annotations

# Compatibility facade for the split HMM decoder modules.
from cw.tools.legacy_decoder.symbol_hmm_decoder import *  # noqa: F401,F403
from cw.tools.legacy_decoder.character_hmm_decoder import *  # noqa: F401,F403
from cw.tools.legacy_decoder.hmm_common import *  # noqa: F401,F403
from cw.tools.legacy_decoder import symbol_hmm_decoder as _symbol
from cw.tools.legacy_decoder import character_hmm_decoder as _char
from cw.tools.legacy_decoder import hmm_common as _common

for _impl in (_symbol, _char, _common):
    globals().update({
        name: value
        for name, value in _impl.__dict__.items()
        if not (name.startswith("__") and name.endswith("__"))
    })

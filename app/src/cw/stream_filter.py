from __future__ import annotations

from cw.stream_models import StreamTrackResult, StreamingConfig, channel_merge_hz


def filter_final_tracks(
    tracks: list[StreamTrackResult],
    config: StreamingConfig,
) -> list[StreamTrackResult]:
    """Return final stream tracks that look publishable.

    Streaming carrier tracking is intentionally sensitive, so it may produce
    Morse-like side interpretations around a strong carrier.  This final gate
    keeps normal good decodes, drops very low quality decodes, and suppresses
    close-by shadow tracks only when they are clearly worse than a neighbouring
    track.  The raw tracker remains available through lower-level debug state;
    the public stream result should avoid presenting these shadows as real
    channels.
    """

    if not tracks:
        return []

    quality_filtered = _quality_filter(tracks, config)
    return _suppress_shadow_tracks(quality_filtered, config)


def _quality_filter(
    tracks: list[StreamTrackResult],
    config: StreamingConfig,
) -> list[StreamTrackResult]:
    max_score = config.max_final_score
    if max_score is None:
        return list(tracks)
    return [track for track in tracks if track.quality.score <= max_score]


def _suppress_shadow_tracks(
    tracks: list[StreamTrackResult],
    config: StreamingConfig,
) -> list[StreamTrackResult]:
    if not tracks:
        return []

    shadow_hz = config.shadow_suppression_hz
    if shadow_hz is None:
        shadow_hz = channel_merge_hz(config)
    if shadow_hz <= 0:
        return list(tracks)

    kept: list[StreamTrackResult] = []
    for track in sorted(tracks, key=lambda item: (item.quality.score, -item.hits)):
        better_neighbour = next(
            (
                existing
                for existing in kept
                if abs(existing.carrier_hz - track.carrier_hz) < shadow_hz
                and track.quality.score >= existing.quality.score + config.shadow_score_margin
            ),
            None,
        )
        if better_neighbour is not None:
            continue
        kept.append(track)

    kept.sort(key=lambda item: item.track_id)
    return kept

from __future__ import annotations


def _format_config(config) -> str:
    return (
        f"f{config.frame_ms:g}/h{config.hop_ms:g}/"
        f"b{config.bandwidth_hz:g}/t{config.threshold_ratio:g}"
    )


def _display_text(text: str) -> str:
    return text or "<empty>"


def _display_track_text(track) -> str:
    if getattr(track, "sessions", None) and len(track.sessions) > 1:
        return " | ".join(
            f"[{session.session_id}] {_display_text(session.decoded.text)}"
            for session in track.sessions
        )
    return _display_text(track.decoded.text)


def _has_multiple_sessions(tracks) -> bool:
    return any(len(getattr(track, "sessions", [])) > 1 for track in tracks)


def _format_update_line(update) -> str:
    return (
        f"t={update.time_s:>7.3f}s "
        f"track={update.track_id:<2} "
        f"session={update.session_id:<2} "
        f"carrier={update.carrier_hz:>7.1f}Hz "
        f"score={update.score:>6.1f} "
        f"text={_display_text(update.text)}"
    )


def _format_event_line(event) -> str:
    session = "-" if event.session_id is None else str(event.session_id)
    reason = f" reason={event.reason}" if event.reason else ""
    text = f" text={_display_text(event.text)}" if event.text else ""
    score = f" score={event.score:.1f}" if event.score is not None else ""
    return (
        f"t={event.time_s:>7.3f}s "
        f"channel={event.channel_id:<2} "
        f"session={session:<2} "
        f"carrier={event.carrier_hz:>7.1f}Hz "
        f"kind={event.kind}"
        f"{score}{reason}{text}"
    )


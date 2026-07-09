from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from cw.receiving.models import ChannelSignal
from cw.signal.config import SignalConfig
from cw.signal.analysis import baseband_envelope, envelope_energy_frames
from cw.signal.models import SignalRun, SignalState, SignalTrack


class SignalSegmenter(Protocol):
    name: str

    def segment(self, channel: ChannelSignal) -> tuple[SignalTrack, ...]: ...


@dataclass(frozen=True)
class _EnergyFrames:
    energy: np.ndarray
    hop_s: float


class DistributionSignalSegmenter:
    """Segments activity from the channel's own energy distribution.

    The model fits two Gaussian components to log-energy values.  The lower
    mean component is SPACE, the higher mean component is MARK.  The configured
    probabilities do not define a fixed energy threshold; they define how sure
    the model must be before it emits MARK or SPACE.  Frames below that certainty
    become UNKNOWN.
    """

    name = "energy_distribution"

    def __init__(self, config: SignalConfig) -> None:
        self.config = config

    def segment(self, channel: ChannelSignal) -> tuple[SignalTrack, ...]:
        frames = _energy_frames(channel, self.config)
        if frames is None:
            return ()
        model = _fit_two_component_log_energy_model(
            frames.energy,
            max_iterations=self.config.signal_distribution_max_iterations,
        )
        if model is None:
            return tuple(
                _all_unknown_track(f"{self.name}:p={probability:.2f}", len(frames.energy), frames.hop_s)
                for probability in self.config.signal_distribution_acceptance_probabilities
            )

        tracks: list[SignalTrack] = []
        for probability in self.config.signal_distribution_acceptance_probabilities:
            activity = _distribution_activity_states(model.mark_probability, acceptance_probability=probability)
            runs = _clean_signal_runs(_runs_from_activity_states(activity, frames.hop_s), self.config)
            tracks.append(
                SignalTrack(
                    analyzer=f"{self.name}:p={probability:.2f}",
                    runs=tuple(runs),
                    unknown_ratio=_unknown_ratio_from_runs(runs),
                )
            )
        return tuple(tracks)


class SignalSegmenterBank:
    def __init__(self, segmenters: tuple[SignalSegmenter, ...]) -> None:
        self.segmenters = segmenters

    @classmethod
    def default(cls, config: SignalConfig) -> "SignalSegmenterBank":
        return cls((DistributionSignalSegmenter(config),))

    def segment_channel(self, channel: ChannelSignal) -> tuple[SignalTrack, ...]:
        tracks: list[SignalTrack] = []
        for segmenter in self.segmenters:
            config = self._config(segmenter)
            for track in segmenter.segment(channel):
                gated = _gate_signal_track(track, config)
                if gated is not None:
                    tracks.append(gated)
        return tuple(tracks)

    @staticmethod
    def _config(segmenter: SignalSegmenter) -> SignalConfig:
        config = getattr(segmenter, "config", None)
        if config is None:
            raise RuntimeError("signal segmenter does not expose its SignalConfig")
        return config


@dataclass(frozen=True)
class _DistributionModel:
    mark_probability: np.ndarray


def _energy_frames(channel: ChannelSignal, config: SignalConfig) -> _EnergyFrames | None:
    if not channel.has_audio:
        return None
    envelope = baseband_envelope(
        channel.audio_window,
        channel.sample_rate,
        channel.carrier_hz,
        lowpass_ms=max(5.0, config.signal_frame_ms / 2.5),
    )
    energy, _frame_times = envelope_energy_frames(envelope, channel.sample_rate, hop_ms=config.signal_hop_ms)
    if len(energy) == 0 or float(np.max(energy)) <= 0:
        return None
    energy = energy.astype(np.float64, copy=False)
    if not _has_keyed_cw_activity(energy, config):
        return None
    return _EnergyFrames(energy=energy, hop_s=config.signal_hop_ms / 1000)


def _fit_two_component_log_energy_model(energy: np.ndarray, *, max_iterations: int) -> _DistributionModel | None:
    if len(energy) < 4:
        return None
    positive_floor = max(float(np.max(energy)) * 1e-12, 1e-18)
    values = np.log(np.maximum(energy.astype(np.float64, copy=False), positive_floor))
    if not np.all(np.isfinite(values)):
        return None
    if float(np.max(values) - np.min(values)) < 1e-6:
        return None

    means = np.asarray([np.percentile(values, 25), np.percentile(values, 75)], dtype=np.float64)
    variance = max(float(np.var(values)), 1e-6)
    variances = np.asarray([variance, variance], dtype=np.float64)
    weights = np.asarray([0.5, 0.5], dtype=np.float64)

    for _ in range(max_iterations):
        log_probs = _component_log_probabilities(values, means, variances, weights)
        responsibilities = _normalize_log_probabilities(log_probs)
        component_weight = np.sum(responsibilities, axis=0)
        if float(np.min(component_weight)) <= 1e-9:
            return None
        weights = component_weight / float(len(values))
        means = np.sum(responsibilities * values[:, None], axis=0) / component_weight
        deviations = values[:, None] - means[None, :]
        variances = np.sum(responsibilities * deviations * deviations, axis=0) / component_weight
        variances = np.maximum(variances, 1e-6)

    log_probs = _component_log_probabilities(values, means, variances, weights)
    responsibilities = _normalize_log_probabilities(log_probs)
    mark_component = int(np.argmax(means))
    space_component = 1 - mark_component
    if float(means[mark_component] - means[space_component]) < 1e-6:
        return None
    return _DistributionModel(mark_probability=responsibilities[:, mark_component])


def _component_log_probabilities(
    values: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    safe_variances = np.maximum(variances, 1e-12)
    safe_weights = np.maximum(weights, 1e-12)
    return (
        np.log(safe_weights)[None, :]
        - 0.5 * np.log(2.0 * np.pi * safe_variances)[None, :]
        - ((values[:, None] - means[None, :]) ** 2) / (2.0 * safe_variances[None, :])
    )


def _normalize_log_probabilities(log_probs: np.ndarray) -> np.ndarray:
    max_log = np.max(log_probs, axis=1, keepdims=True)
    exp_shifted = np.exp(log_probs - max_log)
    sums = np.sum(exp_shifted, axis=1, keepdims=True)
    return exp_shifted / np.maximum(sums, 1e-12)


def _distribution_activity_states(mark_probability: np.ndarray, *, acceptance_probability: float) -> list[SignalState]:
    states: list[SignalState] = []
    for probability in mark_probability:
        probability = float(probability)
        if probability >= acceptance_probability:
            states.append(SignalState.MARK)
        elif (1.0 - probability) >= acceptance_probability:
            states.append(SignalState.SPACE)
        else:
            states.append(SignalState.UNKNOWN)
    return states


def _all_unknown_track(analyzer: str, frame_count: int, hop_s: float) -> SignalTrack:
    duration_s = round(float(max(hop_s, frame_count * hop_s)), 6) if frame_count > 0 else 0.0
    runs = (SignalRun(state=SignalState.UNKNOWN, duration_s=duration_s),) if duration_s > 0 else ()
    return SignalTrack(analyzer=analyzer, runs=runs, unknown_ratio=_unknown_ratio_from_runs(list(runs)))


def _runs_from_activity_states(states: list[SignalState], hop_s: float) -> list[SignalRun]:
    if not states:
        return []
    output: list[SignalRun] = []
    current = states[0]
    length = 1
    for state in states[1:]:
        if state is current:
            length += 1
            continue
        output.append(SignalRun(state=current, duration_s=round(float(max(hop_s, length * hop_s)), 6)))
        current = state
        length = 1
    output.append(SignalRun(state=current, duration_s=round(float(max(hop_s, length * hop_s)), 6)))
    return output



def _clean_signal_runs(runs: list[SignalRun], config: SignalConfig) -> list[SignalRun]:
    """Remove signal-layer micro glitches before decoding.

    The only public control for this physical cleanup is ``signal_max_cpm``.
    It defines the fastest plausible CW keying rate; runs shorter than the
    derived minimum timing unit are not independent Morse elements.

    Cleanup order matters: first bridge tiny gaps/unknowns inside MARKs so a
    real but ragged tone is preserved, then absorb isolated too-fast MARK
    spikes into SPACE.
    """
    cleaned = _merge_adjacent_signal_runs(list(runs))
    min_element_s = _min_plausible_mark_s(config)
    if min_element_s <= 0:
        return cleaned

    cleaned = _absorb_short_unknown_runs(cleaned, max_unknown_s=min_element_s)
    cleaned = _merge_short_internal_signal_runs(cleaned, max_gap_s=min_element_s, gap_state=SignalState.SPACE)
    cleaned = _merge_short_internal_signal_runs(cleaned, max_gap_s=min_element_s, gap_state=SignalState.UNKNOWN)

    cleaned = [
        SignalRun(SignalState.SPACE, run.duration_s)
        if run.state is SignalState.MARK and run.duration_s < min_element_s
        else run
        for run in cleaned
    ]
    cleaned = _mark_stuck_tones_unknown(cleaned, config)
    return _merge_adjacent_signal_runs(cleaned)


def _mark_stuck_tones_unknown(runs: list[SignalRun], config: SignalConfig) -> list[SignalRun]:
    max_mark_s = float(config.signal_max_continuous_mark_s)
    if max_mark_s <= 0:
        return runs
    return [
        SignalRun(SignalState.UNKNOWN, run.duration_s)
        if run.state is SignalState.MARK and run.duration_s > max_mark_s
        else run
        for run in runs
    ]


def _min_plausible_mark_s(config: SignalConfig) -> float:
    """Minimum plausible CW MARK duration from the configured max speed.

    Speed is expressed as characters/minute.  Using the standard PARIS timing,
    one character is about ten timing units on average, so 200 cpm corresponds
    to roughly a 30 ms dot.  MARK runs shorter than this are too fast to be
    reliable Morse elements and are treated as signal glitches.
    """
    max_cpm = float(config.signal_max_cpm)
    if max_cpm <= 0:
        return 0.0
    return 60.0 / (max_cpm * 10.0)


def _absorb_short_unknown_runs(runs: list[SignalRun], *, max_unknown_s: float) -> list[SignalRun]:
    if not runs:
        return []
    output: list[SignalRun] = []
    for index, run in enumerate(runs):
        if run.state is not SignalState.UNKNOWN or run.duration_s > max_unknown_s:
            output.append(run)
            continue
        previous_state = output[-1].state if output else None
        next_state = next((candidate.state for candidate in runs[index + 1 :] if candidate.duration_s > 0), None)
        replacement = previous_state or next_state
        if replacement is None or replacement is SignalState.UNKNOWN:
            output.append(run)
        else:
            output.append(SignalRun(replacement, run.duration_s))
    return _merge_adjacent_signal_runs(output)


def _merge_short_internal_signal_runs(runs: list[SignalRun], *, max_gap_s: float, gap_state: SignalState) -> list[SignalRun]:
    if not runs:
        return []
    output: list[SignalRun] = []
    index = 0
    while index < len(runs):
        run = runs[index]
        if run.state is not SignalState.MARK:
            output.append(run)
            index += 1
            continue

        duration_s = run.duration_s
        index += 1
        while (
            index + 1 < len(runs)
            and runs[index].state is gap_state
            and runs[index].duration_s < max_gap_s
            and runs[index + 1].state is SignalState.MARK
        ):
            duration_s += runs[index].duration_s + runs[index + 1].duration_s
            index += 2
        output.append(SignalRun(SignalState.MARK, round(float(duration_s), 6)))
    return output


def _merge_adjacent_signal_runs(runs: list[SignalRun]) -> list[SignalRun]:
    output: list[SignalRun] = []
    for run in runs:
        if run.duration_s <= 0:
            continue
        if output and output[-1].state is run.state:
            previous = output[-1]
            output[-1] = SignalRun(previous.state, round(float(previous.duration_s + run.duration_s), 6))
        else:
            output.append(run)
    return output

def _gate_signal_track(track: SignalTrack, config: SignalConfig) -> SignalTrack | None:
    if track.unknown_ratio > config.signal_max_unknown_ratio:
        return None
    return track


def _unknown_ratio_from_runs(runs: list[SignalRun]) -> float:
    total_s = sum(max(0.0, float(run.duration_s)) for run in runs)
    if total_s <= 0:
        return 0.0
    unknown_s = sum(
        max(0.0, float(run.duration_s))
        for run in runs
        if run.state is SignalState.UNKNOWN
    )
    return round(max(0.0, min(1.0, unknown_s / total_s)), 6)


def _has_keyed_cw_activity(energy: np.ndarray, config: SignalConfig) -> bool:
    """Return true when the channel envelope looks like keyed CW.

    Two checks are intentionally combined here.  The old standardized
    low/high separation catches broad noise and flat tones, but it can be
    fooled by a steady tone whose envelope has a tiny, very consistent ripple:
    the normalized separation can look large even though there is no real
    MARK/SPACE depth.  A CW channel must therefore have both separable clusters
    and a minimum envelope contrast in dB.
    """

    if len(energy) < 8:
        return True
    finite = energy[np.isfinite(energy)]
    if len(finite) < 8:
        return False
    separation = _keying_separation(finite)
    contrast_db = _keying_contrast_db(finite)
    return (
        separation >= float(config.signal_min_keying_separation)
        and contrast_db >= float(config.signal_min_keying_contrast_db)
    )


def _keying_separation(energy: np.ndarray) -> float:
    log_energy = _finite_log_energy(energy)
    if len(log_energy) == 0:
        return 0.0
    p35, p65 = np.percentile(log_energy, [35, 65])
    low = log_energy[log_energy <= p35]
    high = log_energy[log_energy >= p65]
    if len(low) < 2 or len(high) < 2:
        return 0.0
    midpoint = (float(p35) + float(p65)) / 2.0
    mark_fraction = float(np.mean(log_energy >= midpoint))
    if mark_fraction <= 0.01 or mark_fraction >= 0.99:
        return 0.0
    if not _has_cw_scale_marks(log_energy >= midpoint):
        return 0.0
    pooled_sigma = max(((float(np.var(low)) + float(np.var(high))) / 2.0) ** 0.5, 1e-9)
    return float((float(np.mean(high)) - float(np.mean(low))) / pooled_sigma)


def _keying_contrast_db(energy: np.ndarray) -> float:
    """Robust on/off depth of the keyed envelope in dB.

    This is deliberately not normalized by cluster variance.  A fixed tone with
    small amplitude ripple can score well on normalized separation, but its
    absolute p95-p5 depth stays small compared to real keyed CW.
    """

    log_energy = _finite_log_energy(energy)
    if len(log_energy) == 0:
        return 0.0
    p5, p95 = np.percentile(log_energy, [5, 95])
    return float((10.0 / np.log(10.0)) * max(0.0, float(p95 - p5)))


def _finite_log_energy(energy: np.ndarray) -> np.ndarray:
    if len(energy) == 0:
        return np.asarray([], dtype=np.float64)
    finite = energy[np.isfinite(energy)]
    if len(finite) == 0:
        return np.asarray([], dtype=np.float64)
    max_energy = float(np.max(finite))
    if max_energy <= 0:
        return np.asarray([], dtype=np.float64)
    floor = max(max_energy * 1e-12, 1e-18)
    log_energy = np.log(np.maximum(finite.astype(np.float64, copy=False), floor))
    return log_energy[np.isfinite(log_energy)]


def _has_cw_scale_marks(mark_states: np.ndarray) -> bool:
    # One second of hiss can have high/low envelope samples, but the high runs
    # are usually only a few hop frames long.  A real keyed CW channel must have
    # at least a couple of tone runs long enough to be Morse elements.
    long_mark_runs = 0
    current = bool(mark_states[0]) if len(mark_states) else False
    length = 0
    for state in mark_states:
        state = bool(state)
        if state == current:
            length += 1
            continue
        if current and length >= 10:
            long_mark_runs += 1
        current = state
        length = 1
    if current and length >= 10:
        long_mark_runs += 1
    return long_mark_runs >= 2

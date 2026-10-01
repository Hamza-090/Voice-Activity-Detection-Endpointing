import os

import pandas as pd
import soundfile as sf
import torch

from silero_vad import load_silero_vad, VADIterator


SAMPLE_RATE = 16000
WINDOW = 512

AUDIO_FILES = [
    "recordings/hamza_en_s01.wav",
    "recordings/hamza_en_s04.wav",
    "recordings/hamza_en_s05.wav",
]

SILENCE_SETTINGS = [100, 300, 500]

ALWAYS_ON_CHUNK_SECONDS = 0.5


def load_audio(path):

    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"{path}: expected {SAMPLE_RATE} Hz, got {sr} Hz"
        )

    return audio.astype("float32")


def get_vad_events(audio, model, silence_ms):

    vad = VADIterator(
        model,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=silence_ms
    )

    events = []

    for start in range(
        0,
        len(audio) - WINDOW + 1,
        WINDOW
    ):

        chunk = audio[start:start + WINDOW]

        event = vad(
            torch.from_numpy(chunk),
            return_seconds=True
        )

        if event:

            if "start" in event:
                events.append(
                    (event["start"], "turn_start")
                )

            if "end" in event:
                events.append(
                    (event["end"], "turn_end")
                )

    vad.reset_states()

    return events


def always_on_calls(duration_seconds):

    return int(
        __import__("math").ceil(
            duration_seconds / ALWAYS_ON_CHUNK_SECONDS
        )
    )


def main():

    print("Loading Silero VAD...")

    model = load_silero_vad()

    print("Model loaded.")
    print()

    results = []

    for path in AUDIO_FILES:

        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue

        audio = load_audio(path)

        duration = len(audio) / SAMPLE_RATE

        baseline_calls = always_on_calls(duration)

        print("=" * 60)
        print(os.path.basename(path))
        print(f"Duration: {duration:.2f}s")
        print(f"Always-on ASR calls: {baseline_calls}")

        for silence_ms in SILENCE_SETTINGS:

            events = get_vad_events(
                audio,
                model,
                silence_ms
            )

            completed_segments = sum(
                1
                for _, event_type in events
                if event_type == "turn_end"
            )

            vad_calls = completed_segments

            reduction = (
                (baseline_calls - vad_calls)
                / baseline_calls
                if baseline_calls
                else 0
            )

            results.append({
                "filename": os.path.basename(path),
                "duration_seconds": duration,
                "min_silence_ms": silence_ms,
                "always_on_calls": baseline_calls,
                "vad_triggered_calls": vad_calls,
                "calls_saved": baseline_calls - vad_calls,
                "reduction_percent": reduction * 100,
            })

            print()
            print(
                f"{silence_ms} ms:"
            )
            print(
                f"  VAD-triggered ASR calls: {vad_calls}"
            )
            print(
                f"  Calls saved: "
                f"{baseline_calls - vad_calls}"
            )
            print(
                f"  Reduction: "
                f"{reduction:.1%}"
            )

    df = pd.DataFrame(results)

    df.to_csv(
        "asr_call_comparison.csv",
        index=False
    )

    print()
    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    print()
    summary = df.groupby("min_silence_ms")[
    ["always_on_calls", "vad_triggered_calls", "calls_saved"]
    ].sum()

    summary["reduction_percent"] = (
    summary["calls_saved"] / summary["always_on_calls"] * 100
    )

    print(summary)
        

    print()
    print("Saved: asr_call_comparison.csv")


if __name__ == "__main__":
    main()
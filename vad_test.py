import os

import numpy as np
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


def load_audio(path):
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"{path}: expected {SAMPLE_RATE} Hz, got {sr} Hz"
        )

    return audio.astype(np.float32)


def run_vad(audio, model, min_silence_ms):

    vad = VADIterator(
        model,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=min_silence_ms
    )

    events = []

    for start in range(
        0,
        len(audio) - WINDOW + 1,
        WINDOW
    ):
        chunk = audio[start:start + WINDOW]
        tensor = torch.from_numpy(chunk)

        event = vad(
            tensor,
            return_seconds=True
        )

        if event:

            if "start" in event:
                events.append(
                    (
                        event["start"],
                        "turn_start"
                    )
                )

            if "end" in event:
                events.append(
                    (
                        event["end"],
                        "turn_end"
                    )
                )

    vad.reset_states()

    return events


def main():

    print("Loading Silero VAD...")
    model = load_silero_vad()
    print("Model loaded.")
    print()

    for audio_path in AUDIO_FILES:

        if not os.path.exists(audio_path):
            print(f"NOT FOUND: {audio_path}")
            print()
            continue

        audio = load_audio(audio_path)

        duration = len(audio) / SAMPLE_RATE

        print("=" * 60)
        print(f"File: {audio_path}")
        print(f"Duration: {duration:.2f}s")
        print("=" * 60)

        for silence_ms in SILENCE_SETTINGS:

            events = run_vad(
                audio,
                model,
                silence_ms
            )

            print()
            print(
                f"min_silence_duration_ms = "
                f"{silence_ms}"
            )

            if not events:
                print("No VAD events detected.")
                continue

            for timestamp, event_type in events:
                print(
                    f"{timestamp:.3f}s -> {event_type}"
                )

        print()


if __name__ == "__main__":
    main()
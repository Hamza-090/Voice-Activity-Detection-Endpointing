import os

import pandas as pd
import soundfile as sf
import torch

from silero_vad import load_silero_vad, VADIterator


SAMPLE_RATE = 16000
WINDOW = 512

SILENCE_SETTINGS = [100, 300, 500]


def load_audio(path):

    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"{path}: expected {SAMPLE_RATE} Hz, got {sr} Hz"
        )

    return audio.astype("float32")


def get_vad_endpoints(audio, model, silence_ms):

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


def main():

    labels = pd.read_csv(
        "human_endpoint_labels.csv"
    )

    print("Loading Silero VAD...")

    model = load_silero_vad()

    print("Model loaded.")
    print()

    results = []

    for _, row in labels.iterrows():

        filename = row["filename"]
        human_end = float(row["human_end"])

        path = os.path.join(
            "recordings",
            filename
        )

        audio = load_audio(path)

        for silence_ms in SILENCE_SETTINGS:

            events = get_vad_endpoints(
                audio,
                model,
                silence_ms
            )

            end_events = [
                timestamp
                for timestamp, event_type in events
                if event_type == "turn_end"
            ]

            # ------------------------------------------------
            # Premature endpoints
            # ------------------------------------------------

            premature = [
                timestamp
                for timestamp in end_events
                if timestamp < human_end
            ]

            premature_count = len(premature)

            # ------------------------------------------------
            # First endpoint AT OR AFTER human endpoint
            # ------------------------------------------------

            after_human = [
                timestamp
                for timestamp in end_events
                if timestamp >= human_end
            ]

            if after_human:

                vad_final_end = after_human[0]

                delay = (
                    vad_final_end - human_end
                )

            else:

                vad_final_end = None
                delay = None

            results.append({
                "filename": filename,
                "min_silence_ms": silence_ms,
                "human_end": human_end,
                "premature_endpoint_count": premature_count,
                "premature_endpoints": ",".join(
                    f"{x:.3f}" for x in premature
                ),
                "final_vad_end": vad_final_end,
                "endpoint_delay_seconds": delay
            })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "endpointing_results.csv",
        index=False
    )

    print()
    print("Saved endpointing_results.csv")
    print()

    print(
        results_df.to_string(index=False)
    )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print()
    print("Mean endpoint delay:")
    print(
        results_df
        .groupby("min_silence_ms")[
            "endpoint_delay_seconds"
        ]
        .mean()
    )

    print()
    print("Total premature endpoints:")
    print(
        results_df
        .groupby("min_silence_ms")[
            "premature_endpoint_count"
        ]
        .sum()
    )


if __name__ == "__main__":
    main()
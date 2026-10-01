import os

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt


SAMPLE_RATE = 16000

AUDIO_FILES = [
    "recordings/hamza_en_s01.wav",
    "recordings/hamza_en_s04.wav",
    "recordings/hamza_en_s05.wav",
]


def load_audio(path):

    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"{path}: expected {SAMPLE_RATE} Hz, got {sr} Hz"
        )

    return audio.astype(np.float32)


def main():

    labels = []

    for path in AUDIO_FILES:

        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue

        audio = load_audio(path)

        time = np.arange(len(audio)) / SAMPLE_RATE

        fig, ax = plt.subplots(figsize=(14, 5))

        ax.plot(time, audio)

        ax.set_title(
            f"Click the HUMAN speech END for:\n{path}"
        )

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")

        ax.grid(True)

        print()
        print("=" * 60)
        print(path)
        print("Listen to the recording and inspect the waveform.")
        print("Click ONCE at the point where the speaker's turn")
        print("actually ends.")
        print("=" * 60)

        clicks = plt.ginput(
            1,
            timeout=-1
        )

        plt.close(fig)

        if not clicks:
            print("No point selected.")
            continue

        human_end = clicks[0][0]

        print(
            f"Human endpoint: {human_end:.3f} seconds"
        )

        labels.append({
            "filename": os.path.basename(path),
            "human_end": human_end
        })

    df = pd.DataFrame(labels)

    df.to_csv(
        "human_endpoint_labels.csv",
        index=False
    )

    print()
    print("Saved human_endpoint_labels.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
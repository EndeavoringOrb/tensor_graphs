"""Create small deterministic image, video, and audio inputs for the example."""

import math
import struct
import subprocess
import wave
from pathlib import Path

from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"


def make_image(path: Path) -> None:
    image = Image.new("RGB", (384, 256), (120, 190, 235))
    draw = ImageDraw.Draw(image)
    draw.ellipse((268, 28, 334, 94), fill=(255, 205, 65))
    draw.polygon([(0, 196), (96, 112), (192, 196)], fill=(70, 125, 92))
    draw.polygon([(98, 196), (228, 94), (384, 196)], fill=(53, 105, 82))
    draw.rectangle((0, 196, 384, 256), fill=(112, 164, 91))
    draw.ellipse((158, 183, 220, 245), fill=(190, 85, 55))
    draw.ellipse((175, 193, 187, 205), fill=(25, 25, 25))
    draw.ellipse((194, 193, 206, 205), fill=(25, 25, 25))
    draw.arc((176, 199, 204, 224), 10, 170, fill=(25, 25, 25), width=3)
    image.save(path)


def make_audio(path: Path) -> None:
    sample_rate = 16_000
    duration_seconds = 2.0
    samples = bytearray()
    for index in range(int(sample_rate * duration_seconds)):
        time_seconds = index / sample_rate
        envelope = min(1.0, time_seconds * 8, (duration_seconds - time_seconds) * 8)
        tone = math.sin(2 * math.pi * 440 * time_seconds)
        sample = int(12_000 * envelope * tone)
        samples.extend(struct.pack("<h", sample))

    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(samples)


def make_video(path: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=skyblue:s=384x256:r=8:d=3",
        "-vf",
        "drawbox=x='mod(t*90,340)':y=174:w=44:h=44:color=orangered:t=fill,"
        "drawbox=x=0:y=218:w=384:h=38:color=yellowgreen:t=fill",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_image(ASSETS / "sample.png")
    make_video(ASSETS / "sample.mp4")
    make_audio(ASSETS / "sample.wav")
    print(f"Wrote sample media to {ASSETS}")


if __name__ == "__main__":
    main()

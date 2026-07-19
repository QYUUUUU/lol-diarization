"""
Etape 1 : Telechargement et conversion audio

Prerequis systeme :
    - yt-dlp  (pip install yt-dlp)
    - ffmpeg  (installe sur ta machine : apt install ffmpeg / brew install ffmpeg)

Usage:
    python 01_download_audio.py "https://www.youtube.com/watch?v=XXXX" -o data/match.wav
"""
import argparse
import subprocess
import sys
from pathlib import Path


def download_and_convert(url: str, output_wav: Path) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    # 1. Telecharge uniquement la piste audio avec yt-dlp
    print(f"[1/2] Telechargement de l'audio depuis {url} ...")
    base = output_wav.with_suffix("")
    subprocess.run(
        [
            "yt-dlp",
            "-x",  # extraction audio uniquement
            "--audio-format", "wav",
            "-o", f"{base}.%(ext)s",
            url,
        ],
        check=True,
    )

    downloaded = output_wav.with_suffix(".wav")
    if downloaded != output_wav:
        downloaded.rename(output_wav)

    # 2. Reformatage en 16kHz mono (format attendu par pyannote / whisper)
    print("[2/2] Reformatage en 16kHz mono ...")
    tmp = output_wav.with_name(output_wav.stem + "_16k.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(output_wav), "-ar", "16000", "-ac", "1", str(tmp)],
        check=True,
    )
    tmp.replace(output_wav)
    print(f"Audio pret : {output_wav}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Telecharge une video YouTube et extrait l'audio en wav 16kHz mono."
    )
    parser.add_argument("url", help="URL de la video YouTube")
    parser.add_argument("-o", "--output", default="data/match.wav", help="Chemin du wav de sortie")
    args = parser.parse_args()

    try:
        download_and_convert(args.url, Path(args.output))
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'execution de {e.cmd[0]} : {e}", file=sys.stderr)
        sys.exit(1)

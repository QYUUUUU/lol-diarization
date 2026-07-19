"""
Etape 1 (bis) : Decoupage + conversion d'un audio LOCAL deja telecharge

A utiliser a la place de 01_download_audio.py quand tu as deja le mp3
(ex: video YouTube contenant plusieurs games, tu ne veux garder que la game 1).

Usage:
    # Garder du debut jusqu'a 45:40 (fin game 1 + commentaire VOD)
    python 00_prepare_local_audio.py data/full_video.mp3 --end 45:40 -o data/game1.wav

    # Garder une plage précise (ex: game 2 commence a 46:10 et finit a 1:32:05)
    python 00_prepare_local_audio.py data/full_video.mp3 --start 46:10 --end 1:32:05 -o data/game2.wav

Le timestamp peut etre au format mm:ss, h:mm:ss, ou un nombre de secondes brut.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def parse_timestamp(value: str) -> float:
    """Convertit 'mm:ss', 'h:mm:ss' ou un nombre de secondes en secondes (float)."""
    value = value.strip().replace(";", ":")  # tolere une faute de frappe frequente (; au lieu de :)
    if ":" not in value:
        return float(value)
    parts = [float(p) for p in value.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def trim_and_convert(input_path: Path, output_path: Path, start: float, end: float | None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-i", str(input_path)]
    if start > 0:
        cmd += ["-ss", str(start)]
    if end is not None:
        duration = end - start
        if duration <= 0:
            raise ValueError(f"La fin ({end}s) doit etre apres le debut ({start}s).")
        cmd += ["-t", str(duration)]
    cmd += ["-ar", "16000", "-ac", "1", str(output_path)]

    print(f"Decoupage {start:.0f}s -> {end if end else 'fin'}s, conversion en 16kHz mono ...")
    subprocess.run(cmd, check=True)
    print(f"Audio pret : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decoupe et convertit un fichier audio local en wav 16kHz mono."
    )
    parser.add_argument("input", help="Chemin du fichier audio source (mp3, wav, m4a, ...)")
    parser.add_argument("-o", "--output", default="data/match.wav", help="Chemin du wav de sortie")
    parser.add_argument("--start", default="0", help="Timestamp de debut (mm:ss, h:mm:ss ou secondes). Defaut: 0")
    parser.add_argument("--end", default=None, help="Timestamp de fin (mm:ss, h:mm:ss ou secondes). Defaut: fin du fichier")
    args = parser.parse_args()

    start_sec = parse_timestamp(args.start)
    end_sec = parse_timestamp(args.end) if args.end else None

    try:
        trim_and_convert(Path(args.input), Path(args.output), start_sec, end_sec)
    except subprocess.CalledProcessError as e:
        print(f"Erreur ffmpeg : {e}", file=sys.stderr)
        sys.exit(1)

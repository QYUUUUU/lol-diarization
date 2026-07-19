"""
Etape 2c : Fusion diarisation + transcription

Attribue a chaque segment de texte transcrit le locuteur qui parle le plus
pendant ce segment (overlap maximal entre segment Whisper et segments pyannote).

Usage:
    python 04_merge.py data/diarization.json data/transcription.json -o data/merged.json
"""
import argparse
import json
from pathlib import Path


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speaker(seg: dict, diarization: list) -> str:
    best_speaker, best_overlap = "UNKNOWN", 0.0
    for turn in diarization:
        ov = overlap(seg["start"], seg["end"], turn["start"], turn["end"])
        if ov > best_overlap:
            best_overlap, best_speaker = ov, turn["speaker"]
    return best_speaker


def merge(diarization: list, transcription: list) -> list:
    merged = []
    for seg in transcription:
        speaker = assign_speaker(seg, diarization)
        merged.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": speaker,
            "text": seg["text"],
        })
    return merged


def format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusionne diarisation et transcription en sous-titres attribues.")
    parser.add_argument("diarization", help="Chemin vers diarization.json")
    parser.add_argument("transcription", help="Chemin vers transcription.json")
    parser.add_argument("-o", "--output", default="data/merged.json")
    args = parser.parse_args()

    diarization = json.loads(Path(args.diarization).read_text())
    transcription = json.loads(Path(args.transcription).read_text())

    merged = merge(diarization, transcription)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2))

    print(f"{len(merged)} lignes fusionnees sauvegardees dans {out_path}\n")
    print("Apercu :")
    for seg in merged[:10]:
        print(f"[{format_timestamp(seg['start'])}] {seg['speaker']} : {seg['text']}")

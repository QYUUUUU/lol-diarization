"""
Etape 2b : Transcription (Speech-to-Text) avec OpenAI Whisper

Prerequis :
    pip install openai-whisper
    (necessite aussi ffmpeg, deja requis a l'etape 1)

Usage:
    python 03_transcribe.py data/match.wav -o data/transcription.json --model small --lang fr
"""
import argparse
import json
from pathlib import Path
from typing import Optional


def run_transcription(wav_path: Path, model_size: str, language: Optional[str]) -> list:
    import whisper  # import local : chargement du modele est lourd

    print(f"Chargement du modele Whisper '{model_size}' ...")
    model = whisper.load_model(model_size)

    print(f"Transcription de {wav_path} ...")
    result = model.transcribe(str(wav_path), language=language, word_timestamps=True, verbose=False)

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        })

    print(f"{len(segments)} segments transcrits.")
    return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcrit un fichier audio avec Whisper.")
    parser.add_argument("wav", help="Chemin du fichier wav")
    parser.add_argument("-o", "--output", default="data/transcription.json")
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--lang", default="fr", help="Code langue (fr, en, ...) ou 'auto' pour detection")
    args = parser.parse_args()

    language = None if args.lang == "auto" else args.lang
    segments = run_transcription(Path(args.wav), args.model, language)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2))
    print(f"Resultat sauvegarde dans {out_path}")

"""
Etape 2a : Diarisation (separation des locuteurs) avec pyannote.audio

Prerequis :
    - Un compte HuggingFace + token (https://hf.co/settings/tokens)
    - Avoir accepte les conditions d'utilisation de :
        https://huggingface.co/pyannote/speaker-diarization-3.1
        https://huggingface.co/pyannote/segmentation-3.0
    - export HF_TOKEN="hf_xxxxx"   (ou --token en argument)

Usage:
    python 02_diarize.py data/match.wav -o data/diarization.json
    python 02_diarize.py data/match.wav -o data/diarization.json --num-speakers 5
"""
import argparse
import json
import os
from pathlib import Path
from typing import Optional


def run_diarization(wav_path: Path, hf_token: str, num_speakers: Optional[int] = None) -> list:
    from pyannote.audio import Pipeline  # import local : lourd, charge seulement si besoin

    # Selon la version de pyannote.audio installee, le parametre s'appelle
    # soit `token` (versions recentes), soit `use_auth_token` (versions plus anciennes).
    # On essaie les deux pour ne pas dependre d'une version precise.
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )

    print(f"Diarisation de {wav_path} (peut prendre plusieurs minutes a plusieurs heures sur CPU selon la duree du fichier) ...")
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}

    try:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        with ProgressHook() as hook:
            diarization = pipeline(str(wav_path), hook=hook, **kwargs)
    except ImportError:
        # Selon la version de pyannote.audio, ProgressHook peut ne pas etre disponible.
        # On continue sans barre de progression plutot que de planter.
        diarization = pipeline(str(wav_path), **kwargs)

    segments = []
    if hasattr(diarization, "itertracks"):
        # Ancien format (pyannote.audio 3.x) : objet Annotation avec .itertracks()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker,
            })
    elif hasattr(diarization, "speaker_diarization"):
        # Nouveau format (pyannote.audio 4.x) : objet DiarizeOutput avec .speaker_diarization
        for turn, speaker in diarization.speaker_diarization:
            segments.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker,
            })
    else:
        raise TypeError(
            f"Format de sortie pyannote non reconnu : {type(diarization)}. "
            "Verifie la version installee de pyannote.audio."
        )

    n_speakers = len(set(s["speaker"] for s in segments))
    print(f"{len(segments)} segments de parole detectes, {n_speakers} locuteurs distincts.")
    return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diarisation d'un fichier audio avec pyannote.audio.")
    parser.add_argument("wav", help="Chemin du fichier wav (16kHz mono recommande)")
    parser.add_argument("-o", "--output", default="data/diarization.json")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                         help="Token HuggingFace (ou variable d'env HF_TOKEN)")
    parser.add_argument("--num-speakers", type=int, default=None,
                         help="Nombre de locuteurs si connu (accelere et fiabilise le resultat)")
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("Aucun token HuggingFace fourni. Utilise --token ou export HF_TOKEN=...")

    segments = run_diarization(Path(args.wav), args.token, args.num_speakers)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2))
    print(f"Resultat sauvegarde dans {out_path}")
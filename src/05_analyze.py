"""
Etape 3 : Analyse statistique et data visualisation

Genere :
    - out/speaking_time.html   : temps de parole total par speaker (barres)
    - out/comm_density.html    : densite de communication dans le temps, events superposes
    - out/speaking_time.csv, out/comm_density.csv : donnees brutes exportees

Usage:
    python 05_analyze.py data/diarization.json data/merged.json data/events_example.json -o out/
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def load_json(path: str) -> list:
    return json.loads(Path(path).read_text())


def speaking_time_per_speaker(diarization: list) -> pd.DataFrame:
    df = pd.DataFrame(diarization)
    df["duration"] = df["end"] - df["start"]
    grouped = df.groupby("speaker", as_index=False)["duration"].sum()
    grouped = grouped.sort_values("duration", ascending=False)
    return grouped


def word_density_over_time(merged: list, bin_seconds: int = 10) -> pd.DataFrame:
    rows = []
    for seg in merged:
        n_words = len(seg["text"].split())
        mid_time = (seg["start"] + seg["end"]) / 2
        rows.append({"time": mid_time, "n_words": n_words})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["bin"] = (df["time"] // bin_seconds) * bin_seconds
    density = df.groupby("bin", as_index=False)["n_words"].sum()
    density = density.rename(columns={"bin": "time", "n_words": "words_per_bin"})
    return density


def plot_speaking_time(df: pd.DataFrame, out_path: Path) -> None:
    fig = go.Figure(go.Bar(x=df["speaker"], y=df["duration"], marker_color="#5B8DEF"))
    fig.update_layout(
        title="Temps de parole total par joueur",
        xaxis_title="Speaker",
        yaxis_title="Temps de parole (secondes)",
        template="plotly_white",
    )
    fig.write_html(out_path)
    print(f"Graphique genere : {out_path}")


def plot_comm_density(density: pd.DataFrame, events: list, out_path: Path) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=density["time"], y=density["words_per_bin"],
        mode="lines", line=dict(color="#5B8DEF", width=2),
        name="Mots / 10s",
    ))

    y_max = density["words_per_bin"].max() if not density.empty else 1

    for event in events:
        fig.add_vline(x=event["time"], line_width=2, line_dash="dash", line_color="crimson")
        fig.add_annotation(
            x=event["time"], y=y_max, text=event["event"],
            showarrow=False, yshift=15, textangle=-90,
            font=dict(size=10, color="crimson"),
        )

    fig.update_layout(
        title="Densite de communication vs evenements de la partie",
        xaxis_title="Temps (secondes)",
        yaxis_title="Nombre de mots (par tranche de 10s)",
        template="plotly_white",
    )
    fig.write_html(out_path)
    print(f"Graphique genere : {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse et visualise les donnees fusionnees.")
    parser.add_argument("diarization", help="Chemin vers diarization.json")
    parser.add_argument("merged", help="Chemin vers merged.json")
    parser.add_argument("events", help="Chemin vers events.json")
    parser.add_argument("-o", "--outdir", default="out/")
    args = parser.parse_args()

    diarization = load_json(args.diarization)
    merged = load_json(args.merged)
    events = load_json(args.events)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    speaking_df = speaking_time_per_speaker(diarization)
    density_df = word_density_over_time(merged)

    plot_speaking_time(speaking_df, outdir / "speaking_time.html")
    plot_comm_density(density_df, events, outdir / "comm_density.html")

    speaking_df.to_csv(outdir / "speaking_time.csv", index=False)
    density_df.to_csv(outdir / "comm_density.csv", index=False)
    print("Analyse terminee.")

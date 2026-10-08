"""Evaluate VoxScribe: Word Error Rate, response time and command accuracy.

Manifest CSV columns (header required):
    audio,reference,condition[,expected_action]

  audio            path to a WAV/FLAC file
  reference        what was actually said
  condition        label such as quiet, noisy, accent-1
  expected_action  optional: command action the clip should trigger
                   (stop, save_note, clear, copy, download, settings, search)

Usage:  python evaluate.py data/manifest.csv [--no-denoise]
"""
import argparse
import csv
import time
from collections import defaultdict

import jiwer
import soundfile as sf

from commands import match_command, normalise
from preprocessing import TARGET_SR, preprocess, to_mono_16k
from recognizer import Recognizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--no-denoise", action="store_true")
    args = parser.parse_args()

    recognizer = Recognizer()
    refs, hyps = defaultdict(list), defaultdict(list)
    times = defaultdict(list)
    cmd_total = cmd_ok = 0

    with open(args.manifest, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            audio, sr = sf.read(row["audio"], dtype="float32")
            start = time.perf_counter()
            audio = preprocess(to_mono_16k(audio, sr), TARGET_SR, not args.no_denoise)
            text = recognizer.transcribe(audio)
            elapsed = time.perf_counter() - start

            cond = row.get("condition") or "all"
            refs[cond].append(normalise(row["reference"]))
            hyps[cond].append(normalise(text))
            times[cond].append(elapsed)

            expected = (row.get("expected_action") or "").strip()
            if expected:
                cmd_total += 1
                found = match_command(text)
                cmd_ok += int(bool(found) and found["action"] == expected)

    print("| Condition | Clips | WER (%) | Mean time (s) |")
    print("|---|---|---|---|")
    all_refs, all_hyps = [], []
    for cond in refs:
        wer = jiwer.wer(refs[cond], hyps[cond]) * 100
        mean_t = sum(times[cond]) / len(times[cond])
        print(f"| {cond} | {len(refs[cond])} | {wer:.1f} | {mean_t:.2f} |")
        all_refs += refs[cond]
        all_hyps += hyps[cond]
    if all_refs:
        print(f"| overall | {len(all_refs)} | {jiwer.wer(all_refs, all_hyps) * 100:.1f} | |")
    if cmd_total:
        print(f"\nCommand accuracy: {cmd_ok}/{cmd_total} = {100 * cmd_ok / cmd_total:.1f}%")


if __name__ == "__main__":
    main()

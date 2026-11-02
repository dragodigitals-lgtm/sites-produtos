#!/usr/bin/env python3
"""Original instrumental track generator for wwwswipe reels (royalty-free: composed in code).

Usage: python3 music.py OUT.wav --seconds 12 --seed 7 [--style lofi|ambient|drive]
Each seed picks a key, tempo and chord progression, so every reel gets a different track.
"""
import argparse, random
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

SR = 44100
PROGRESSIONS = [  # semitone offsets of chord roots in a minor key
    [0, 8, 3, 10],   # i - VI - III - VII
    [0, 5, 8, 7],    # i - iv - VI - V
    [0, 10, 8, 10],  # i - VII - VI - VII
    [0, 3, 8, 5],    # i - III - VI - iv
]
STYLES = {"lofi": (78, 88), "ambient": (68, 76), "drive": (100, 112)}


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def env(n, a=0.01, r=0.3):
    t = np.arange(n) / SR
    e = np.minimum(1, t / a) * np.exp(-t / r)
    return e


def lowpass(x, cut):
    b, a = butter(2, cut / (SR / 2))
    return lfilter(b, a, x)


def chord_tones(root, minor_quality):
    third = 3 if minor_quality else 4
    return [root, root + third, root + 7, root + 12 + (2 if random.random() < .5 else 0)]


def pad(freqs, n):
    t = np.arange(n) / SR
    out = np.zeros(n)
    for f in freqs:
        for det in (-0.12, 0.0, 0.12):  # detuned voices = warm pad
            ph = random.random() * 6.28
            out += np.sin(2 * np.pi * f * (1 + det / 100) * t + ph) + 0.25 * np.sin(4 * np.pi * f * t + ph)
    a = np.minimum(1, t / 0.6) * np.minimum(1, (t[-1] - t) / 0.4 + 0.05)
    return out * a / (len(freqs) * 3)


def kick(n):
    t = np.arange(n) / SR
    f = 50 + 90 * np.exp(-t * 35)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)


def hat(n):
    return np.random.randn(n) * np.exp(-np.arange(n) / SR * 60) * 0.25


def snare(n):
    t = np.arange(n) / SR
    return (np.random.randn(n) * 0.5 + 0.4 * np.sin(2 * np.pi * 190 * t)) * np.exp(-t * 18)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--seconds", type=float, default=12)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--style", default=None, choices=list(STYLES))
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)
    style = a.style or random.choice(list(STYLES))
    bpm = random.randint(*STYLES[style])
    key = random.randint(45, 52)  # A2..E3 region
    prog = random.choice(PROGRESSIONS)
    beat = 60 / bpm
    total = int(a.seconds * SR)
    mix = np.zeros(total + SR)
    bar = int(4 * beat * SR)

    # chords + bass
    i, pos = 0, 0
    while pos < total:
        root = key + prog[i % 4]
        minor_q = prog[i % 4] in (0, 5)
        notes = chord_tones(root + 12, minor_q)
        seg = pad([midi(n) for n in notes], bar)
        mix[pos:pos + bar] += 0.55 * seg[: len(mix) - pos]
        for b in range(4):  # bass on beats 1 and 3 (+ syncopation for drive)
            if b in (0, 2) or (style == "drive"):
                s = pos + int(b * beat * SR); n = int(beat * SR * 0.9)
                t = np.arange(n) / SR
                bs = np.sin(2 * np.pi * midi(root) * t) * env(n, 0.005, 0.35)
                mix[s:s + n] += 0.5 * bs[: max(0, len(mix) - s)]
        pos += bar; i += 1

    # arpeggio sparkle (pentatonic minor), not in ambient
    if style != "ambient":
        scale = [0, 3, 5, 7, 10, 12, 15]
        step = beat / 2
        k = 0
        while k * step * SR < total:
            if random.random() < 0.55:
                s = int(k * step * SR); n = int(step * SR * 1.6)
                t = np.arange(n) / SR
                f = midi(key + 24 + random.choice(scale))
                tone = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * env(n, 0.003, 0.25)
                mix[s:s + n] += 0.12 * tone[: max(0, len(mix) - s)]
            k += 1

    # drums
    if style != "ambient":
        nb = int(a.seconds / beat) + 1
        for b in range(nb):
            s = int(b * beat * SR)
            if s >= total: break
            if b % 4 in (0, 2) or (style == "drive"):
                k_ = kick(int(0.4 * SR)); mix[s:s + len(k_)] += 0.6 * k_[: len(mix) - s]
            if b % 4 in (1, 3):
                sn = snare(int(0.25 * SR)); mix[s:s + len(sn)] += 0.18 * lowpass(sn, 5000)[: len(mix) - s]
            for half in (0, 0.5):
                hs = s + int(half * beat * SR)
                h = hat(int(0.08 * SR)); mix[hs:hs + len(h)] += 0.12 * h[: max(0, len(mix) - hs)]

    mix = mix[:total]
    if style == "lofi":
        mix = lowpass(mix, 4200) + np.random.randn(total) * 0.004  # warm + vinyl hiss
    # fades + normalize
    fi, fo = int(0.3 * SR), int(1.2 * SR)
    mix[:fi] *= np.linspace(0, 1, fi); mix[-fo:] *= np.linspace(1, 0, fo)
    mix = np.tanh(mix / (np.max(np.abs(mix)) + 1e-9) * 1.2) * 0.8
    st = np.stack([mix, np.roll(mix, 220)], axis=1)  # slight stereo width
    wavfile.write(a.out, SR, (st * 32767).astype(np.int16))
    print(f"style={style} bpm={bpm} key={key} prog={prog}")


if __name__ == "__main__":
    main()

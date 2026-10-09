"""Regenerates the figures and the numbers shown in the README.

    pip install numpy scipy matplotlib
    python docs/figures/make_figures.py

Uses the AR coefficients and helpers of data.py and the same synthesis steps as Ej2.py
(audio playback is not needed, so sounddevice/soundfile are stubbed out).
"""
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.io import wavfile
from scipy.signal import freqz, lfilter, spectrogram

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for f in Path("/usr/share/fonts/lm").glob("lm*10-*.otf"):  # Latin Modern, if installed
    font_manager.fontManager.addfont(str(f))
plt.style.use(HERE / "paper.mplstyle")
C = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for mod in ("sounddevice", "soundfile"):  # only used for playback in data.py
    sys.modules.setdefault(mod, types.ModuleType(mod))
sys.path.insert(0, str(ROOT))
import data  # noqa: E402

FS = 14700
VOWELS, FRICATIVES = ["a", "e", "i", "o", "u"], ["sh", "f", "s", "j"]
PHONEMES = VOWELS + FRICATIVES
N = int(0.2 * FS)  # 200 ms analysis window, as in Ej1.py


def save(fig, name):
    fig.savefig(HERE / name, metadata={"Date": None})
    plt.close(fig)


def segment(p):
    rate, x = wavfile.read(ROOT / f"{p}.wav")
    assert rate == FS
    seg = x[:N].astype(float)
    return seg / np.max(np.abs(seg))


def periodogram_db(seg):
    P = np.abs(np.fft.rfft(seg)) ** 2 / len(seg)
    return np.fft.rfftfreq(len(seg), 1 / FS), 10 * np.log10(P + 1e-12)


def ar_envelope_db(p, freqs):
    a = [1.0] + [-ai for ai in data.coef_a[p]]
    _, H = freqz(data.coef_b[p], a, worN=freqs, fs=FS)
    return 10 * np.log10(np.abs(H) ** 2 + 1e-20)


def autocorr(seg):
    r = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
    return r / r[0]


def pitch(seg):  # strongest autocorrelation peak for lags of 2.5-16.7 ms (60-400 Hz)
    r = autocorr(seg)
    lo, hi = int(FS / 400), int(FS / 60)
    k = lo + np.argmax(r[lo:hi])
    return FS / k, r[k]


# ---- Figure 1: recorded /a/ and /s/ against the AR model
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0), gridspec_kw={"width_ratios": [1, 1, 1.25]})
for row, (p, label) in enumerate((("a", "vowel /a/"), ("s", "fricative /s/"))):
    seg = segment(p)
    t = np.arange(N) / FS * 1e3
    win = (t >= 100) & (t < 130)
    axes[row, 0].plot(t[win], seg[win], color=C[0], lw=0.8)
    r = autocorr(seg)
    lag = np.arange(N) / FS * 1e3
    axes[row, 1].plot(lag[lag < 25], r[lag < 25], color=C[0], lw=0.8)
    axes[row, 1].axhline(0, color="#8c8c8c", lw=0.5)
    f, Pdb = periodogram_db(seg)
    env = ar_envelope_db(p, f)
    env += Pdb.max() - env.max()  # align peaks, as in the report
    axes[row, 2].plot(f / 1e3, Pdb - Pdb.max(), color="#9a9a9a", lw=0.5, label="periodogram")
    axes[row, 2].plot(f / 1e3, env - Pdb.max(), color=C[1], lw=1.2, label="AR(20) envelope")
    axes[row, 2].set_ylim(-100, 5)
    axes[row, 0].set_ylabel(label)
    if row == 0:
        f0, rho = pitch(seg)
        axes[row, 1].axvline(1e3 / f0, color=C[1], ls="--", lw=0.8)
        axes[row, 1].annotate(f"$T_0$ = {1e3 / f0:.2f} ms", (1e3 / f0, rho), xytext=(7, 0.85),
                              fontsize=8, color=C[1], bbox=dict(fc="white", ec="none", pad=0.5))
for j, title in enumerate(("(a) Waveform, 100–130 ms", "(b) Normalized autocorrelation",
                           "(c) Spectrum (dB re peak)")):
    axes[0, j].set_title(title)
axes[1, 0].set_xlabel("time [ms]")
axes[1, 1].set_xlabel("lag [ms]")
axes[1, 2].set_xlabel("frequency [kHz]")
axes[1, 2].legend(loc="lower left")
fig.tight_layout()
save(fig, "fig1-analysis.svg")

# ---- Figure 2: AR envelope vs periodogram, all nine phonemes
fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.4), sharex=True, sharey=True)
for ax, p in zip(axes.flat, PHONEMES):
    f, Pdb = periodogram_db(segment(p))
    env = ar_envelope_db(p, f)
    env += Pdb.max() - env.max()
    ax.plot(f / 1e3, Pdb - Pdb.max(), color="#9a9a9a", lw=0.4)
    ax.plot(f / 1e3, env - Pdb.max(), color=C[1] if p in VOWELS else C[0], lw=1.1)
    ax.text(0.96, 0.9, f"/{p}/", transform=ax.transAxes, ha="right", va="top")
    ax.set_ylim(-95, 5)
for ax in axes[-1]:
    ax.set_xlabel("frequency [kHz]")
for ax in axes[:, 0]:
    ax.set_ylabel("dB re peak")
fig.tight_layout(h_pad=0.4, w_pad=0.4)
save(fig, "fig2-envelopes.svg")

# ---- Figure 3: synthesized sequence with variable pitch (Ej2.py, second sequence)
DUR = int(FS * 500 / 1000)


def synth(p, f0=200):  # identical steps to sintetizar_fonema() in Ej2.py
    u = data.gen_pulsos(f0, DUR, FS) if p in VOWELS else np.random.default_rng(0).standard_normal(DUR)
    x = lfilter(data.coef_b[p], [1.0] + [-ai for ai in data.coef_a[p]], u)
    x = data.suavizar_bordes(x, 0.2)
    return x / (np.max(np.abs(x)) + 1e-8)


PITCH = {"a": 100, "e": 125, "i": 150, "o": 125, "u": 100}
seq = np.concatenate([synth(p, PITCH.get(p, 200)) for p in PHONEMES])
t = np.arange(len(seq)) / FS
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 3.6), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1.5]})
ax1.plot(t, seq, color=C[0], lw=0.3, rasterized=True)
ax1.set_ylim(-1.15, 1.6)
ax1.set_ylabel("amplitude")
for i, p in enumerate(PHONEMES):
    lab = f"/{p}/\n{PITCH[p]} Hz" if p in PITCH else f"/{p}/\nnoise"
    ax1.text((i + 0.5) * DUR / FS, 1.12, lab, ha="center", va="bottom", fontsize=7.5, linespacing=0.95)
    for ax in (ax1, ax2):
        ax.axvline(i * DUR / FS, color="#8c8c8c", lw=0.5, ls=":")
ax1.set_title("(a) Waveform")
fq, ts, S = spectrogram(seq, fs=FS, window="hann", nperseg=368, noverlap=276)
Sdb = 10 * np.log10(S + 1e-12)
ax2.pcolormesh(ts, fq / 1e3, Sdb, cmap="Greys", vmin=np.percentile(Sdb, 20), vmax=np.percentile(Sdb, 99.5),
               shading="auto", rasterized=True)
ax2.set_ylabel("frequency [kHz]")
ax2.set_xlabel("time [s]")
ax2.set_title("(b) Spectrogram (25 ms Hann window)")
ax2.tick_params(top=False, right=False)
fig.tight_layout()
save(fig, "fig3-synthesis.svg")

# ---- numbers for Table 1
print("| Phoneme | Excitation | F0 [Hz] | rho(T0) | max |pole| |")
for p in PHONEMES:
    f0, rho = pitch(segment(p))
    poles = np.abs(np.roots([1.0] + [-ai for ai in data.coef_a[p]]))
    print(f"| /{p}/ | {'pulses' if p in VOWELS else 'noise'} | {f0:.0f} | {rho:.2f} | {poles.max():.3f} |")

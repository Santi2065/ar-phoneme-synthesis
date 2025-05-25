import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import sounddevice as sd
import data

# Parámetros
fs = 14700
dur_ms = 500
dur_samples = int(fs * dur_ms / 1000)
transicion = 0.2
vocales = ['a', 'e', 'i', 'o', 'u']
consonantes = ['sh', 'f', 's', 'j']
fonemas = vocales + consonantes

# Construir vectores b y a
b_dict = {p: data.coef_b[p] for p in fonemas}
a_dict = {p: [1.0] + [-ai for ai in data.coef_a[p]] for p in fonemas}

# Síntesis de un fonema con lfilter(b,a,u)
def sintetizar_fonema(p, pitch=200):
    if p in vocales:
        u = data.gen_pulsos(pitch, dur_samples, fs)
    else:
        u = np.random.default_rng(0).standard_normal(dur_samples)
    b = b_dict[p]
    a = a_dict[p]
    x = lfilter(b, a, u)
    x = data.suavizar_bordes(x, transicion)
    # Normalización opcional
    peak = np.max(np.abs(x)) + 1e-8
    return x / peak

# 1) Punto (a): sintetizar y guardar señales
senales_fonemas = {}
os.makedirs('output/ej2', exist_ok=True)
plt.figure(figsize=(12, 8))
for i, p in enumerate(fonemas):
    x = sintetizar_fonema(p)  # pitch default 200 Hz
    senales_fonemas[p] = x
    t = np.arange(int(0.2 * fs)) / fs
    plt.subplot(3, 3, i+1)
    plt.plot(t, x[:len(t)])
    plt.title(f'Fonema {p} (200 Hz)')
    plt.xlabel('Tiempo [s]')
plt.tight_layout()
plt.savefig('output/ej2/fonemas.png', dpi=300)
plt.close()

# 2) Punto (b): concatenar usando las señales de (a), pitch fijo 200 Hz
seq1 = np.concatenate([senales_fonemas[p] for p in fonemas])
sd.play(seq1, fs)
sd.wait()

# Guardar gráfica de la señal concatenada
plt.figure(figsize=(12, 4))
t1 = np.arange(len(seq1)) / fs
plt.plot(t1, seq1)
for i, p in enumerate(fonemas):
    pos = i * dur_samples
    plt.axvline(pos / fs, linestyle='--', alpha=0.5)
    plt.text((pos + dur_samples/2) / fs, 0.9, p, ha='center')
plt.title('Señal concatenada 9 fonemas (200 Hz)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.tight_layout()
plt.savefig('output/ej2/señal_concatenada_200hz.png', dpi=300)
plt.close()

# 3) Punto (c): concatenar con pitches variables
pitch_map = {'a': 100, 'e': 125, 'i': 150, 'o': 125, 'u': 100}
seq2 = np.concatenate([
    sintetizar_fonema(p, pitch_map[p]) if p in vocales else senales_fonemas[p]
    for p in fonemas
])
sd.play(seq2, fs)
sd.wait()

# Guardar gráfica de la señal concatenada con pitches variables
plt.figure(figsize=(12, 4))
t2 = np.arange(len(seq2)) / fs
plt.plot(t2, seq2)
for i, p in enumerate(fonemas):
    pos = i * dur_samples
    label = f"{p}\n({pitch_map[p]} Hz)" if p in pitch_map else p
    plt.axvline(pos / fs, linestyle='--', alpha=0.5)
    plt.text((pos + dur_samples/2) / fs, 0.9, label, ha='center')
plt.title('Señal concatenada pitches variables')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.tight_layout()
plt.savefig('output/ej2/señal_concatenada_pitches_variables.png', dpi=300)
plt.close()
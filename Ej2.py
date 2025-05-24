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
vowels = ['a', 'e', 'i', 'o', 'u']
consonants = ['sh', 'f', 's', 'j']

# Construir vectores b y a
b_dict = {p: data.coef_b[p] for p in vowels + consonants}
a_dict = {p: [1.0] + [-ai for ai in data.coef_a[p]] for p in vowels + consonants}

# Síntesis de un fonema con filter(b,a,u)
def sintetizar_fonema(p, pitch=200):
    # Fuente de excitación
    if p in vowels:
        u = data.gen_pulsos(pitch, dur_samples, fs)
    else:
        rng = np.random.default_rng(0)
        u = rng.standard_normal(dur_samples)
    # Filtrado ARMA (solo AR)
    b = b_dict[p]
    a = a_dict[p]
    x = lfilter(b, a, u)
    # Suavizar bordes
    x = data.suavizar_bordes(x, transicion)
    return x

# Graficar primeros 200 ms
os.makedirs('output/ej2', exist_ok=True)
plt.figure(figsize=(12, 8))
for i, p in enumerate(vowels + consonants):
    x = sintetizar_fonema(p)
    t = np.arange(int(0.2 * fs)) / fs
    plt.subplot(3, 3, i+1)
    plt.plot(t, x[:len(t)])
    plt.title(f'Fonema {p}')
    plt.xlabel('Tiempo [s]')
plt.tight_layout()
plt.savefig('output/ej2/fonemas.png')
plt.close()

# Concatenar y reproducir pitch fijo
seq1 = np.concatenate([sintetizar_fonema(p) for p in vowels + consonants])
sd.play(seq1, fs)
sd.wait()

# Concatenar con pitches variables
pitch_map = {'a':100,'e':125,'i':150,'o':125,'u':100}
seq2 = np.concatenate(
    [sintetizar_fonema(p, pitch_map[p]) if p in vowels else sintetizar_fonema(p)
     for p in vowels + consonants]
)
sd.play(seq2, fs)
sd.wait()
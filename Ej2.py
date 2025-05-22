import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from data import coef_a, coef_b, gen_pulsos, suavizar_bordes

fs = 14700
dur_ms = 500
dur_samples = int(fs * dur_ms / 1000)
transicion = 0.2
fonemas_vocales = ['a', 'e', 'i', 'o', 'u']
fonemas_consonantes = ['sh', 'f', 's', 'j']

def sintetizar_fonema(fonema, pitch=None):
    a = coef_a[fonema]
    b = coef_b[fonema][0]
    orden = len(a)
    
    if fonema in fonemas_vocales:
        f0 = pitch if pitch else 200
        U = gen_pulsos(f0, dur_samples, fs)
    else:
        U = np.random.normal(0, 1, dur_samples)
    
    X = np.zeros(dur_samples)
    for n in range(orden, dur_samples):
        X[n] = np.dot(a, X[n-orden:n][::-1]) + b * U[n]
    
    X = suavizar_bordes(X, transicion)
    return X

#a
plt.figure(figsize=(12, 8))
for i, fonema in enumerate(fonemas_vocales + fonemas_consonantes):
    X = sintetizar_fonema(fonema)
    plt.subplot(3, 3, i+1)
    plt.plot(np.arange(0, int(0.2*fs)) / fs, X[:int(0.2*fs)])
    plt.title(f"Fonema {fonema}")
    plt.xlabel("Tiempo [s]")
plt.tight_layout()
plt.savefig("output/ej2/fonemas.png")
plt.show()

#b
secuencia_1 = np.concatenate([sintetizar_fonema(f) for f in fonemas_vocales + fonemas_consonantes])
sd.play(secuencia_1, samplerate=fs)
sd.wait()

#c
pitch_por_fonema = {'a': 100, 'e': 125, 'i': 150, 'o': 125, 'u': 100}
secuencia_2 = []
for f in fonemas_vocales:
    secuencia_2.append(sintetizar_fonema(f, pitch=pitch_por_fonema[f]))
for f in fonemas_consonantes:
    secuencia_2.append(sintetizar_fonema(f))
secuencia_2 = np.concatenate(secuencia_2)
sd.play(secuencia_2, samplerate=fs)
sd.wait()

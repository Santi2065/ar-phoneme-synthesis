import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import freqz
import data

# Parámetros
fs = 14700
phonemes = ['a', 'e', 'i', 'o', 'u', 'sh', 'f', 's', 'j']
output_dir = 'output/ej1'
os.makedirs(output_dir, exist_ok=True)

for p in phonemes:
    # Cargo señal
    rate, x = wavfile.read(f'{p}.wav')
    assert rate == fs, f"Frecuencia de muestreo inesperada en {p}.wav"

    # Tomo los primeros 200 ms
    N = int(0.2 * fs)
    seg = x[:N]

    # Autocorrelación normalizada
    ac = np.correlate(seg, seg, mode='full')
    ac_pos = ac[len(ac)//2:]
    lags = np.arange(len(ac_pos)) / fs
    ac_norm = ac_pos / np.max(np.abs(ac_pos))

    plt.figure()
    plt.plot(lags, ac_norm)
    plt.title(f'{p} – Autocorrelación normalizada')
    plt.xlabel('Retraso [s]')
    plt.ylabel('Autocorrelación')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{p}_autocorr.png')
    plt.close()

    # Periodograma
    nfft = N
    X = np.fft.rfft(seg, n=nfft)
    Pxx = np.abs(X)**2 / N
    freqs = np.fft.rfftfreq(nfft, d=1/fs)
    Pxx_db = 10 * np.log10(Pxx + 1e-12)

    # PSD teórica usando freqz
    a_coeffs = data.coef_a[p]
    b_coeffs = data.coef_b[p]

    b = b_coeffs
    a = [1.0] + [-ai for ai in a_coeffs]
    w, H = freqz(b, a, worN=nfft, fs=fs)
    H2_db = 20 * np.log10(np.abs(H) + 1e-12)

    # PSD de excitación
    if p in ['sh', 'f', 's', 'j']:
        Su = np.ones_like(w)
    else:
        f_p, Su_p = data.psd_pulsos(f0=200, N=N, fs=fs)
        Su = np.interp(w, f_p, Su_p)
    Su_db = 10 * np.log10(Su + 1e-12)

    # Graficar periodograma y PSD teórica
    plt.figure()
    plt.plot(freqs, Pxx_db, label='Periodograma empírico')
    plt.plot(w, H2_db + Su_db, label='PSD teórica')
    plt.title(f'{p} – Periodograma vs PSD teórica')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Magnitud [dB]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{p}_psd.png')
    plt.close()

print("¡Listo! Gráficos en 'output/ej1/'")
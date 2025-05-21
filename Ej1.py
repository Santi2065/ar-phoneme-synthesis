import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import periodogram

import data

fs = 14700
phonemes = ['a', 'e', 'i', 'o', 'u', 'sh', 'f', 's', 'j']
output_dir = 'output/ej1'
os.makedirs(output_dir, exist_ok=True)

for p in phonemes:
    wav_path = f'{p}.wav'
    rate, x = wavfile.read(wav_path)
    assert rate == fs, f"Frecuencia de muestreo inesperada en {wav_path}"
    N = int(0.2 * fs)
    seg = x[:N]
    t = np.arange(N) / fs

    # Señal en tiempo
    plt.figure()
    plt.plot(t, seg)
    plt.title(f'{p} – Señal (200 ms)')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{p}_time.png')
    plt.close()

    # Autocorrelación
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

    # Periodograma empírico
    f, Pxx = periodogram(seg, fs)
    Pxx_db = 10 * np.log10(Pxx + 1e-12)

    # Cálculo de H2 = |H(f)|^2
    a_coeffs = data.coef_a[p]
    b_coeffs = data.coef_b[p]

    den = 1 - sum(
        a_coeffs[i] * np.exp(-2j * np.pi * f * (i+1) / fs)
        for i in range(len(a_coeffs))
    )
    num = sum(
        b_coeffs[k] * np.exp(-2j * np.pi * f * k / fs)
        for k in range(len(b_coeffs))
    )
    H2 = np.abs(num)**2 / (np.abs(den)**2)

    # PSD teórica: H2 * S_U(f)
    if p in ['a', 'e', 'i', 'o', 'u']:
        # psd_pulsos(f0, N, fs)
        f_p, Su_p = data.psd_pulsos(200, N, fs)
        Su = np.interp(f, f_p, Su_p)
    else:
        Su = np.ones_like(f)

    SX_db = 10 * np.log10(H2 * Su + 1e-12)

    # Graficar periodograma vs PSD teórica
    plt.figure()
    plt.plot(f, Pxx_db, label='Periodograma empírico')
    plt.plot(f, SX_db, label='PSD teórica')
    plt.title(f'{p} – Periodograma vs PSD teórica')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Magnitud [dB]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{p}_psd.png')
    plt.close()

print("Gráficos generados en 'output/ej1/'")
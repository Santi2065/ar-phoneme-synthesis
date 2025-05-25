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
    # 1) Cargo WAV y tomo 200 ms
    rate, x = wavfile.read(f'{p}.wav')
    assert rate == fs, f"Se esperaba {fs} Hz en {p}.wav, pero vino {rate} Hz"
    N = int(0.2 * fs)
    seg = x[:N].astype(float)

    # Normalizar señal a [-1,1]
    max_val = np.max(np.abs(seg))
    if max_val > 0:
        seg /= max_val

    # Eje de tiempo en segundos
    t = np.arange(N) / fs

    # Autocorrelación normalizada
    ac_full = np.correlate(seg, seg, mode='full')
    ac = ac_full[N-1:] / np.max(np.abs(ac_full[N-1:]))
    lags = np.arange(len(ac)) / fs

    # Periodograma empírico (FFT)
    X = np.fft.rfft(seg, n=N)
    Pxx = np.abs(X)**2 / N
    freqs = np.fft.rfftfreq(N, d=1/fs)
    Pxx_db = 10 * np.log10(Pxx + 1e-12)

    # PSD teórica del modelo AR
    a_coeffs = data.coef_a[p]
    b_coeffs = data.coef_b[p]
    b = b_coeffs
    a = [1.0] + [-ai for ai in a_coeffs]
    w, H = freqz(b, a, worN=N, fs=fs)
    H2 = np.abs(H)**2

    if p in ['sh','f','s','j']:
        Su = np.ones_like(w)
    else:
        f_p, Su_p = data.psd_pulsos(f0=200, N=N, fs=fs)
        Su = np.interp(w, f_p, Su_p)

    SX = H2 * Su
    SX_db = 10 * np.log10(SX + 1e-12)

    # Normalizar PSD para comparar formas
    Pxx_db_norm = Pxx_db - np.max(Pxx_db)
    SX_db_norm = SX_db - np.max(SX_db)

    # Panel 1×3: señal, autocorr, PSD normalizadas
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)

    # (a) Señal en el dominio del tiempo
    axs[0].plot(t, seg)
    axs[0].set(title=f'{p} – Señal (200 ms)',
               xlabel='Tiempo [s]',
               ylabel='Amplitud normalizada')

    # (b) Autocorrelación
    axs[1].plot(lags, ac)
    axs[1].set(title='Autocorrelación',
               xlabel='Retraso [s]',
               ylabel='Autocorrelación normalizada')

    # (c) Periodograma vs PSD
    axs[2].plot(freqs, Pxx_db_norm, label='Empírico')
    axs[2].plot(w, SX_db_norm, label='Teórica')
    axs[2].set(title='Periodograma vs PSD',
               xlabel='Frecuencia [Hz]',
               ylabel='Magnitud [dB]')
    axs[2].legend(fontsize='small')

    # Guardar panel
    plt.savefig(f'{output_dir}/{p}_panel.png', dpi=150)
    plt.close(fig)

print(f"Paneles generados en '{output_dir}'")

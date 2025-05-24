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
    # Cargo y recorto 200 ms
    rate, x = wavfile.read(f'{p}.wav')
    assert rate == fs, f"Se esperaba {fs} Hz, pero {rate} Hz en {p}.wav"
    N = int(0.2 * fs)
    seg = x[:N]

    # Dominio del tiempo
    t = np.arange(N) / fs

    # Autocorrelación normalizada
    ac_full = np.correlate(seg, seg, mode='full')
    ac = ac_full[N-1:] / np.max(np.abs(ac_full[N-1:]))
    lags = np.arange(len(ac)) / fs

    # Periodograma con FFT de NumPy
    X = np.fft.rfft(seg, n=N)
    Pxx = (np.abs(X)**2) / N
    freqs = np.fft.rfftfreq(N, d=1/fs)
    Pxx_db = 10 * np.log10(Pxx + 1e-12)

    # PSD teórica usando freqz
    a_coeffs = data.coef_a[p]
    b_coeffs = data.coef_b[p]
    b = b_coeffs
    a = [1.0] + [-ai for ai in a_coeffs]
    w, H = freqz(b, a, worN=N, fs=fs)
    H2 = np.abs(H)**2

    # PSD de excitación
    if p in ['sh', 'f', 's', 'j']:
        Su = np.ones_like(w)
    else:
        f_p, Su_p = data.psd_pulsos(f0=200, N=N, fs=fs)
        Su = np.interp(w, f_p, Su_p)

    SX = H2 * Su
    SX_db = 10 * np.log10(SX + 1e-12)

    # Panel 1×3
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)
    axs[0].plot(t, seg)
    axs[0].set(title=f'{p} – Señal (200 ms)', xlabel='Tiempo [s]')
    axs[1].plot(lags, ac)
    axs[1].set(title='Autocorrelación', xlabel='Retraso [s]')
    axs[2].plot(freqs, Pxx_db, label='Empírico')
    axs[2].plot(w, SX_db,     label='Teórica')
    axs[2].set(title='Periodograma vs PSD', xlabel='Frecuencia [Hz]')
    axs[2].legend(fontsize='small')

    plt.savefig(f'{output_dir}/{p}_panel.png', dpi=150)
    plt.close(fig)

print(f"Paneles generados en '{output_dir}'")
import numpy as np
from scipy.fft import fft
from scipy.signal import freqz, lfilter
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd

# Coeficiente b
coef_b = {'a':[0.0311], 'e':[0.0321],'i':[0.0121], 'o':[0.0073], 'u': [0.00258], 's':[0.0251], 'sh':[0.0302], 'f':[0.0453], 'j':[0.0245]}

# Coeficientes ai
coef_a = {
        "a":[1.1835, -0.5685, 0.3874, 0.2939, -0.9807, 0.587, -0.2441, -0.0866, 0.4763, -0.0491, -0.1137, 0.0601, -0.4138, 0.2613, 0.1998, -0.0327, 0.0894, -0.195, -0.0695, 0.01],
        "e":[0.3876, -0.8134, 1.126, 0.3371, 0.7794, 0.413, -0.6026, -0.4977, -0.9342, -0.3311, -0.0233, 0.3248, 0.388, 0.4025, 0.0977, -0.0062, -0.0465, -0.1614, -0.0132, -0.1101],
        "i":[-0.2189, -0.6421, 1.3317, 1.3899, 1.6527, 0.3503, -1.0505, -1.8235, -1.6228, -0.5322, 0.3435, 0.9051, 0.7906, 0.3556, 0.2841, -0.164, 0.084, -0.2809, -0.0976, -0.1635],
        "o":[1.2328, -0.6982, 1.4335, -0.4589, 0.3123, -1.1168, -0.6169, -0.1906, 0.5552, 0.5243, 0.8605, -0.0488, -0.3003, -0.855, -0.1109, -0.0894, 0.5002, 0.0034, 0.3986, -0.386],
        "u":[0.735, 0.3731, 0.5123, 1.0994, -1.1137, -1.1504, -0.3682, -0.3828, 0.7703, 0.8673, 0.2769, -0.198, -0.1659, -0.3783, -0.1847, 0.1825, 0.1184, -0.0033, 0.1232, -0.1316],
        "s":[-0.3646, 0.1339, -0.0423, -0.1338, 0.1461, 0.0452, 0.2571, 0.2483, 0.055, 0.0955, 0.265, 0.204, -0.0319, -0.1486, 0.0459, 0.0725, 0.1031, -0.0649, -0.0051, 0.0186],
        "sh":[1.0213, -1.6453, 0.9882, -1.4537, 0.9707, -1.0667, 0.9042, -0.7057, 0.8852, -0.5809, 0.7712, -0.3084, 0.4782, -0.3091, 0.4741, -0.3427, 0.3427, -0.2138, 0.0853, -0.0054],
        "f":[0.5988, -0.0429, -0.3181, -0.2044, 0.3401, -0.2233, -0.0889, 0.1002, 0.0213, -0.0634, 0.0929, 0.1195, -0.1242, 0.1506, -0.0795, 0.1444, 0.0262, 0.0056, 0.0531, 0.0727],
        "j":[1.1653, -0.6726, 0.67, -0.3186, -0.2648, -0.1825, 0.1804, -0.0515, 0.1982, -0.1682, 0.0735, 0.0452, -0.1658, 0.1665, -0.0253, -0.0508, 0.2067, -0.1849, 0.0751, 0.026]
         }

def gen_pulsos(f0, N, fs):
    """
    Genera un tren de impulsos periodico en el tiempo.
    f0: frecuencia fundamental (pitch) del tren de impulsos [Hz].
    N: cantidad de puntos que posee el array de la secuencia generada.
    fs: frecuencia de muestreo [Hz].
    Retorna: tren de impulsos (con varianza normalizada) de frecuencia f0.
    """
    s = np.zeros(N)
    s[np.arange(N) % round(fs / f0) == 0] = np.sqrt(fs / f0)
    return s

def psd_pulsos(f0, N, fs):
    """
    Genera la densidad espectral de potencia de un tren de impulsos.
    f0: frecuencia fundamental [Hz] (pitch) del tren de impulsos en el tiempo.
    N: cantidad de puntos que posee el array de la PSD resultante (SU(w)).
    fs: frecuencia de muestreo [Hz].
    Retorna:
    - PSD del tren de impulsos
    - Vector de frecuencias del espectro [Hz]
    """
    u = gen_pulsos(f0, N, fs)
    f = np.arange(N) * fs /N    # Vector de frecuencias (Hz)
    Su = np.abs(fft(u))**2 / N  # Periodograma
    return Su, f

def suavizar_bordes(x, fade=20):
    """
    Suaviza los bordes de una señal.
    x: señal original (array).
    fade: (float) porcentaje de transición en los bordes (0-50% del largo de x)
    retorna: versión suavizada de x
    """
    N = len(x)
    fade = max(1, min(fade, 50))  # Limita fade entre 1 y 50
    M = 2 * int(fade / 100 * N)
    v = np.hamming(M)
    fade_in = v[:M // 2]
    fade_out = v[M // 2:]
    window = np.concatenate([fade_in, np.ones(N - M), fade_out])
    s = window * x
    return s

def reproducir(audio, fs):
    """
    Reproducir audio usando soundevice
    audio: array con el contenido de la señal
    fs: freucencia de muestreo [Hz]
    """
    sd.play(audio, fs)
    sd.wait()

from math import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

from scipy.signal import resample
from scipy.signal.windows import hann
from scipy.linalg import solve_toeplitz, toeplitz

# -----------------------------------------------------------------------------
# Block decomposition
# -----------------------------------------------------------------------------


def blocks_decomposition(x, w, R=0.5):
    """
    Performs the windowing of the signal

    Parameters
    ----------

    x: numpy array
      single channel signal
    w: numpy array
      window
    R: float (default: 0.5)
      overlapping between subsequent windows

    Return
    ------

    out: (blocks, windowed_blocks)
      block decomposition of the signal:
      - blocks is a list of the audio segments before the windowing
      - windowed_blocks is a list the audio segments after windowing
    """

    # Padding
    nperseg = len(w)
    noverlap = int(R*len(w))
    hop_length = nperseg - noverlap
    x_padded = np.pad(
        x, (nperseg // 2, nperseg // 2), "constant", constant_values=(0, 0)
    )

    # Block decomposition
    blocks, windowed_blocks = [], []
    offset = 0
    while offset <= x.size:
        block = x_padded[offset : offset + nperseg]
        blocks.append(block)
        windowed_blocks.append(block * w)
        offset += hop_length

    return (np.array(blocks), np.array(windowed_blocks))


def blocks_reconstruction(blocks, w, signal_size, R=0.5):
    """
    Reconstruct a signal from overlapping blocks

    Parameters
    ----------

    blocks: numpy array
      signal segments. blocks[i,:] contains the i-th windowed
      segment of the speech signal
    w: numpy array
      window
    signal_size: int
      size of the original signal
    R: float (default: 0.5)
      overlapping between subsequent windows

    Return
    ------

    out: numpy array
      reconstructed signal
    """
    nperseg = w.size
    noverlap = int(R*len(w))
    hop_length = nperseg - noverlap

    reconstruction = np.zeros(signal_size + nperseg)
    norm = np.zeros_like(reconstruction)
    offset = 0
    for block in blocks:
        reconstruction[offset : offset + nperseg] += block * w
        norm[offset : offset + nperseg] += w * w
        offset += hop_length

    reconstruction = reconstruction[nperseg // 2 : -nperseg // 2]
    norm = norm[nperseg // 2 : -nperseg // 2]

    return reconstruction / norm


# -----------------------------------------------------------------------------
# Linear Predictive coding
# -----------------------------------------------------------------------------


def autocovariance(x, k):
    """
    Estimates the autocovariance C[k] of signal x

    Parameters
    ----------

    x: numpy array
      speech segment to be encoded
    k: int
      covariance index
    """
    N = len(x)
    r = 0
    for i in range(N - k):
        r += x[i] * x[i + k]
    return r / N


def lpc_encode(x, p):
    """
    Linear predictive coding

    Predicts the coefficient of the linear filter used to describe the
    vocal track

    Parameters
    ----------

    x: numpy array
      segment of the speech signal
    p: int
      number of coefficients in the filter

    Returns
    -------

    out: tuple (coef, prediction)
      coefs: numpy array
        filter coefficients
      prediction: numpy array
        lpc prediction
    """
    N = len(x)
    prediction = np.zeros(N)

    rs0 = np.array([autocovariance(x, i) for i in range(p)])
    rs1 = np.array([autocovariance(x, i) for i in range(1, p + 1)])
    coef = solve_toeplitz(rs0, rs1)

    for n in range(p):
        prediction[n] = np.sum(coef[:n+1] * x[n:: -1])
    for n in range(p,N):
        prediction[n] = np.sum(coef * x[n:n-p: -1])
    return (coef, prediction)


def lpc_decode(coefs, source):
    """
    Synthesizes a speech segment using the LPC filter and an excitation source

    Parameters
    ----------

    coefs: numpy array
      filter coefficients

    source: numpy array
      excitation signal

    Returns
    -------

    out: numpy array
      synthesized segment
    """

    # A COMPLETER
    return 0


def estimate_pitch(signal, sample_rate, min_freq=50, max_freq=200, threshold=1):
    """
    Estimate the pitch of an audio segment using the autocorrelation method and
    indicate whether or not it is a voiced signal

    Parameters
    ----------

    signal: array-like
      audio segment
    sample_rate: int
      sample rate of the audio signal
    min_freq: int
      minimum frequency to consider (default 50 Hz)
    max_freq: int
      maximum frequency to consider (default 200 Hz)
    threshold: float
      threshold used to determine whether or not the audio segment is voiced

    Returns
    -------

    voiced: boolean
      Indicates if the signal is voiced (True) or not
    pitch: float
      estimated pitch (in s)
    """

    # A COMPLETER
    return 0

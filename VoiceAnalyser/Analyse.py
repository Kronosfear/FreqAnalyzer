# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 23:30:21 2016

@author: Vijay
"""

from scipy.io.wavfile import read
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.signal import get_window
from math import ceil
import numpy
from pylab import figure, imshow, clf, gray, xlabel, ylabel

# Read in a wav file 
#   returns sample rate (samples / sec) and data
rate, data = read('demo.wav')

# Define the sample spacing and window size.
dT = 1.0/rate
T_window = 50e-3
N_window = int(T_window * rate)
N_data = len(data)

# 1. Get the window profile
window = get_window('hamming', N_window)

# 2. Set up the FFT
result = []
start = 0
while (start < N_data - N_window):
    end = start + N_window
    result.append(fftshift(fft(window*data[start:end])))
    start = end

result.append(fftshift(fft(window*data[-N_window:])))
result = numpy.array(result,result[0].dtype)

# Display results
freqscale = fftshift(fftfreq(N_window,dT))[150:-150]/4e3
figure(1)
clf()
imshow(abs(result[:,150:-150]),extent=(freqscale[-1],freqscale[0],(N_data*dT-T_window/2.0),T_window/2.0))
xlabel('Frequency (kHz)')
ylabel('Time (sec.)')
gray()
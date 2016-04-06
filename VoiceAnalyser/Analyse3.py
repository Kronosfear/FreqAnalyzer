from pylab import plot, show, title, xlabel, ylabel, subplot, savefig
from scipy import fft, arange, ifft
from scipy.signal import blackmanharris, fftconvolve, kaiser
from numpy import sin, linspace, pi, diff, log, argmax
from numpy.fft import rfft
from matplotlib.mlab import find
import numpy
from scipy.io.wavfile import read,write
import wave
import struct
import scipy.fftpack
import math
import pyaudio


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)



def FFTSpectrum(y, Fs):
    n = len(y)
    k = arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(int(n/2))] # one side frequency range
    Y = fft(y)/n # fft computing and normalization
    Y = Y[range(int(n/2))]
    plot(frq,abs(Y),'r') # plotting the spectrum
    xlabel('Freq (Hz)')
    ylabel('|Y(freq)|')


def TimeAmpSpectrum(filename):
    rate,data=read(filename)
    y=data[:]
    l=len(y)
    timp=len(y)/44100.
    t=linspace(0,timp,len(y))
    subplot(2,1,1)
    plot(t,y)
    xlabel('Time')
    ylabel('Amplitude')
    subplot(2,1,2)
    FFTSpectrum(y,Fs)
    show()




def fft_cepstral(filename):
    wave_file = wave.open(filename, 'r')
    nframes = wave_file.getnframes()
    nchannels = wave_file.getnchannels()
    sampling_frequency = wave_file.getframerate()
    T = nframes / float(sampling_frequency)
    read_frames = wave_file.readframes(nframes)
    wave_file.close()
    datas = struct.unpack("%dh" %  nchannels*nframes, read_frames)
    datas = numpy.array(datas)
    w = numpy.fft.fft(datas)
    
    ceps=ifft(w);
    posmax = ceps.argmax();
    result = Fs/(32*nframes*nchannels)*(posmax-1) 
    print("Cepstral: ", result)
    
    freqs = numpy.fft.fftfreq(len(w))
    idx = numpy.argmax(numpy.abs(w))
    freq = freqs[idx]
    freq_in_hertz = abs(freq * Fs)
    print("FFT: ", freq_in_hertz)


def zerocross(filename):
    sig=read(filename)
    y= numpy.array(sig)
    indices = find((sig[1:] >= 0) & (sig[:-1] < 0))
    crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in indices]
    Frequency =  fs / mean(diff(crossings))
    print ("Zero Crossing:  ",Frequency)

    
    
Fs = 11025;  # sampling rate

filename = 'male/demo.wav'
fft_cepstral(filename)

TimeAmpSpectrum(filename)

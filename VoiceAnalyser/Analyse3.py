from pylab import plot, show, title, xlabel, ylabel, subplot, savefig
from scipy import fft, arange, ifft
from scipy.signal import blackmanharris, fftconvolve, kaiser, hamming
from numpy import sin, linspace, pi, diff, log, argmax
from numpy.fft import rfft, irfft
from matplotlib.mlab import find
from array import array
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
    
def normalize(snd_data):
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)
    
    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r




def my_fft(filename):
    wave_file = wave.open(filename, 'r')
    nframes = wave_file.getnframes()
    nchannels = wave_file.getnchannels()
    sampling_frequency = wave_file.getframerate()
    T = nframes / float(sampling_frequency)
    read_frames = wave_file.readframes(nframes)
    wave_file.close()
    datas = struct.unpack("%dh" %  nchannels*nframes, read_frames)
    datas = numpy.array(datas)
    datas = normalize(datas)
    w = numpy.fft.rfft(datas)
    
    freqs = numpy.fft.fftfreq(len(w))
    idx = numpy.argmax(numpy.abs(w))
    freq = freqs[idx]
    freq_in_hertz = abs(freq * sampling_frequency)/2
    print("FFT: ", freq_in_hertz)


def zerocross(filename):
    sig=read(filename)
    indices = find((sig[1:] >= 0) & (sig[:-1] < 0))
    crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in indices]
    Frequency =  fs / mean(diff(crossings))
    print ("Zero Crossing:  ",Frequency)
    
def cepstral(spf):
    index1=15000;
    frameSize=4096;
    spf = wave.open(filename, 'r')
    fs = spf.getframerate();
    signal = spf.readframes(-1);
    signal = numpy.fromstring(signal, 'Int16');
    index2=index1+frameSize-1;
    frames=signal[index1:int(index2)+1]
    
    zeroPaddedFrameSize=16*frameSize;

    frames2=frames*hamming(len(frames));   
    frameSize=len(frames);
    
    if (zeroPaddedFrameSize>frameSize):
        zrs= numpy.zeros(zeroPaddedFrameSize-frameSize);
        frames2=numpy.concatenate((frames2, zrs), axis=0)

    fftResult=numpy.log(numpy.abs(fft(frames2)));
    ceps=ifft(fftResult);
    nceps=ceps.shape[-1]*2/3
    peaks = []
    k=3
    while(k < nceps - 1):
        y1 = (ceps[k - 1])
        y2 = (ceps[k])
        y3 = (ceps[k + 1])
        if (y2 > y1 and y2 >= y3): peaks.append([float(fs)/(k+2),abs(y2), k, nceps])
        k=k+1
    maxi=max(peaks, key = lambda x: x[1])
    print("Cepstral: ", fs/maxi[0])
    

    
    
Fs = 11025;  # sampling rate

filename = 'male/no.wav'
my_fft(filename)
cepstral(filename)
TimeAmpSpectrum(filename)

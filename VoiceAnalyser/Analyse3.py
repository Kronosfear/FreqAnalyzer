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



def plotSpectru(y,Fs):
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
    lungime=len(y)
    timp=len(y)/44100.
    t=linspace(0,timp,len(y))
    subplot(2,1,1)
    plot(t,y)
    xlabel('Time')
    ylabel('Amplitude')
    subplot(2,1,2)
    plotSpectru(y,Fs)
    show()


def zerocrossings(signal, RATE):
    signal = numpy.fromstring(signal, 'Int16');
    crossing = [math.copysign(1.0, s) for s in signal]
    index = find(numpy.diff(crossing));
    f0=round(len(index) * RATE /(2*numpy.prod(len(signal))))
    return f0;


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
    result = Fs/(64*nframes*nchannels)*(posmax-1) 
    print("Cepstral: ", result)
    
    freqs = numpy.fft.fftfreq(len(w))
    idx = numpy.argmax(numpy.abs(w))
    freq = freqs[idx]
    freq_in_hertz = abs(freq * Fs)
    print("FFT: ", freq_in_hertz)


def Pitch(signal):
    signal = np.fromstring(signal, 'Int16');
    crossing = [math.copysign(1.0, s) for s in signal]
    index = find(np.diff(crossing));
    f0=round(len(index) *RATE /(2*np.prod(len(signal))))
    return f0;


def zerocross(filename):
    for i in range(0, RATE / chunk * RECORD_SECONDS):
    data = stream.read(chunk)
    Frequency=Pitch(data)
    print "%f Frequency" %Frequency

    
    
Fs = 11025;  # sampling rate

filename = 'male/no.wav'
fft_cepstral(filename)


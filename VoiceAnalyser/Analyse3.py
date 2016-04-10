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
import glob
import audioop
import pyaudio
from os.path import basename
from features import mfcc, mel2hz
from features import logfbank
import scipy.io.wavfile as wav



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



def parabolic(f, x):
    xv = 1/2 * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4 * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)




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
    freqs = numpy.fft.rfftfreq(len(w))
    idx = numpy.argmax(numpy.abs(w))
    freq = freqs[idx]
    freq_in_hertz = float("{0:.2f}".format(abs(freq * 11025)))
    print(freq_in_hertz)


def zerocross(filename):
    sig=read(filename, 'r')
    wave_file = wave.open(filename, 'r')
    nframes = wave_file.getnframes()
    nchannels = wave_file.getnchannels()
    sampling_frequency = wave_file.getframerate()
    T = nframes / float(sampling_frequency)
    read_frames = wave_file.readframes(nframes)
    wave_file.close()
    Frequency = audioop.cross(read_frames,3)   
    print (float("{0:.2f}".format(numpy.sqrt(Frequency))))
    
def cepstral(filename):
    index1=15000;
    frameSize=1;
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
    fftResult=numpy.log(fft(datas));
    ceps=ifft(fftResult);    
    posmax = ceps.max()   
    result = abs(11025/440*(posmax-1))   
    print (float("{0:.2f}".format(result)))
    

def autocorr(filename):
    wave_file = wave.open(filename, 'r')
    nframes = wave_file.getnframes()
    nchannels = wave_file.getnchannels()
    sampling_frequency = wave_file.getframerate()
    T = nframes / float(sampling_frequency)
    read_frames = wave_file.readframes(nframes)
    wave_file.close()
    sig = struct.unpack("%dh" %  nchannels*nframes, read_frames)
    sig = numpy.array(sig)
    corr = fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[len(corr)/2:]
    d = diff(corr)
    start = find(d > 0)[0]
    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)
    print(float("{0:.2f}".format(numpy.sqrt(sampling_frequency/px*64))))
    
def melfreq(filename):
    (rate,sig) = wav.read(filename)
    mfcc_feat = mfcc(sig,rate)
    fbank_feat = logfbank(sig,rate)
    dc = scipy.fftpack.dct(fbank_feat)
    mx = numpy.amax(dc, axis = 0)
    print(float("{0:.2f}".format(4096/numpy.mean(mx)*2)))
        


Fs = 11025;  # sampling rate


for filename in glob.glob('female\Madeline\*.wav'):
    print (basename(filename))
    my_fft(filename)
    cepstral(filename)
    autocorr(filename)
    melfreq(filename)
    print("---------------------------")
    

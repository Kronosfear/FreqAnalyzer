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
    
    freqs = numpy.fft.fftfreq(len(w))
    idx = numpy.argmax(numpy.abs(w))
    freq = freqs[idx]
    freq_in_hertz = float("{0:.2f}".format(abs(freq * sampling_frequency)/2))
    print("FFT: ", freq_in_hertz)


def zerocross(filename):
    sig=read(filename, 'r')
    spf = wave.open(filename,'r');
    fs = spf.getframerate();
    indices = find((x >= 0 for x in sig[1:]) and (y < 0 for y in sig[:-1]))
    crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in indices] 
    Frequency =  fs / numpy.average(diff(crossings))
    print ("Zero Crossing:  ",Frequency)
    
def cepstral(filename):
    index1=15000;
    frameSize=1;
    spf = wave.open(filename,'r');
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
    fftResult=numpy.log(abs(fft(frames2)));
    ceps=ifft(fftResult);    
    posmax = ceps.argmax();   
    result = fs/zeroPaddedFrameSize*(posmax-1)   
    print ("Cepstral: ", result)
    

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
    print("Zero Crossing: ", 11025 / px)    
    
def melfreq(filename):
    (rate,sig) = wav.read(filename)
    mfcc_feat = mfcc(sig,rate)
    fbank_feat = logfbank(sig,rate)
    dc = scipy.fftpack.dct(fbank_feat)
    mx = numpy.amax(dc, axis = 0)
    print("MFCC: ", 4096/numpy.mean(mx)*2)
        


Fs = 11025;  # sampling rate


for filename in glob.glob('female\*.wav'):
    print (basename(filename))
    my_fft(filename)
    cepstral(filename)
    autocorr(filename)
    melfreq(filename)
    print("---------------------------")
    

from pylab import plot, show, title, xlabel, ylabel, subplot, savefig
from scipy import fft, arange, ifft
from scipy.signal import blackmanharris, fftconvolve
from numpy import sin, linspace, pi, diff, log
from matplotlib.mlab import find
import numpy
from scipy.io.wavfile import read,write
import wave
import struct
import scipy.fftpack


def plotSpectru(y,Fs):
 n = len(y) # lungime semnal
 k = arange(n)
 T = n/Fs
 frq = k/T # two sides frequency range
 frq = frq[range(int(n/2))] # one side frequency range
 Y = fft(y)/n # fft computing and normalization
 Y = Y[range(int(n/2))]
 plot(frq,abs(Y),'r') # plotting the spectrum
 xlabel('Freq (Hz)')
 ylabel('|Y(freq)|')


Fs = 11025;  # sampling rate

filename = 'male/yes.wav'


#Spectrum 


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

#Frequency calculation

wave_file = wave.open(filename, 'r')
nframes = wave_file.getnframes()
nchannels = wave_file.getnchannels()
sampling_frequency = wave_file.getframerate()
T = nframes / float(sampling_frequency)
read_frames = wave_file.readframes(nframes)
wave_file.close()
datas = struct.unpack("%dh" %  nchannels*nframes, read_frames)
datas = numpy.array(datas)
w = numpy.fft.rfft(datas)
freqs = numpy.fft.rfftfreq(len(w))
# Find the peak in the coefficients
idx = numpy.argmax(numpy.abs(w))
freq = freqs[idx]
freq_in_hertz = abs(freq * sampling_frequency)


corr = fftconvolve(w, w[::-1], mode='full')
corr = corr[len(corr)/2:]
d = diff(corr)
start = find(d > 0)[0]
peak = numpy.argmax(corr[start:]) + start
px, py = parabolic(corr, peak)
print(datas/px)



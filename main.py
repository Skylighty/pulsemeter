import numpy as np
import matplotlib.pyplot as plt
import cv2

# Declare means array
means = []

# Declare VideoCapture feature
cap = cv2.VideoCapture("sample8.mp4")

# Loop untill the end of the video - frame by frame operations
while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # Extract shape from frame
    (x, y, colors) = frame.shape

    # Slicing 350x350 square from the center
    x1 = int((x / 2) - 175)
    x2 = int((x / 2) + 175)
    y1 = int((y / 2) - 175)
    y2 = int((y / 2) + 175)
    square = frame[y1:y2, x1:x2, 0]

    # Compute the mean of values in our square and add it to list
    mean = np.mean(square, dtype=np.float64)
    means.append(mean)

cap.release()
cv2.destroyAllWindows()

# Filtering
filter = [-0.02286961, -0.06362756,  0.57310236,  0.57310236, -0.06362756, -0.02286961]
filtered = np.convolve(means, filter, 'same')

# FFT calculation
fourier = abs(np.fft.fft(filtered))
fourier[0] = 0

# Plot the means
fig = plt.figure()
plt.plot(filtered)
plt.show()

# Raw data is ready, let's plot it:
fig2 = plt.figure()
plt.plot(fourier)
plt.show()

# Variables necessary for calculations
fps = 30.0
framecount = len(filtered)
fftmean = np.mean(fourier[:100])
fftpos = []
fftvalues = []

# Create a list of indexes which contain high-value frequencies
for i in range(100):
    if fourier[i] > fftmean:
        fftpos.append(i)
        fftvalues.append(round(fourier[i], 5))

# Create a list of frequencies under previously specified indexes
freqs = []
for i in range(len(fftpos)):
    freqs.append(round((fftpos[i]*fps)/framecount, 5))

# Calculate BPM value for previosuly calculated frequencies
bpms = []
for i in range(len(freqs)):
    bpms.append(round(freqs[i]*60, 5))

# Finding the most proper value basic on coefficients
# (fftamp*bpm)/freq
coefficients = {}
for i in range(len(bpms)):
    if freqs[i] >= 1:
        coefficients[((fftvalues[i]*bpms[i])/freqs[i])] = bpms[i]
maxc = max(coefficients.keys())
print('Pulse that programme found matching is : ' + str(coefficients[maxc]) +'bpm')

# Output - write results to text file
file = open('bpms.txt', 'w')
file.write('FFT pos         Freq[Hz]            FFT Amp                   BPM\n')
for i in range(len(freqs)-1):
    file.write(str(fftpos[i]) + '               ' + str(freqs[i]) + '             '
               + str(fftvalues[i])+'               ' + str(bpms[i])+'\n')
try:
    file.close()
    print('File cotaining the results saved successfully')
except:
    print('An error has occured during saving a file')



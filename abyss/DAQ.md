# DAQ

The DAQ data is recorded from external sensors and data hacked from the tool. The data is recorded at a rate of 1e5 Hz and is stored in TDMS files.

## Structure

TDMS files are organised into groups and channels (signals). Each file has 5 groups organised H1 to H5 and the channels are named PXI1Slot2/a* with the final character being a zero-based index.

The indices correspond to the following signals
 - a0 Torque?
 - a1 Thrust?
 - a2 Acoustic Emission (AE) Unfiltered
 - a3 Acoustic Emission (AE) RMS filtered
 - a4 Vibration X
 - a5 Vibration Z
     + Direction is perpendicular to hole
 - a6 Feed Drive Current
 - a7 Spindle Drive Current

For the Manchester DAQ data, the material stack is CFRP to Aluminium.

![image](https://user-images.githubusercontent.com/46482002/174842301-85edbc23-dfd3-4209-9b03-854454121dbc.png)


## MITIS

MITIS is a vibration management system that introduces an additional dynamic proportional to the target RPM to help improve performance. For the Manchester DAQ data the vibration is set to 2.5 oscillations per revolution. E.g.

```
MITIS = 2.5 osc/rev
RPM = 6000 RPM = 100 rev/sec
... MITIS freq = 100 rev/s * 2.5 = 250 Hz

On a wavelet plot, that would appear around the scale 1e5/250 = 400
```

## Plots + Dynamics

For the plots, I'll be using file UC-1, Group H5.

### a0

![UC - 1-H5-PXI1Slot2-ai0](https://user-images.githubusercontent.com/46482002/169823937-7a593616-342e-4387-bb2d-b98bbdd3212f.png)

### a1

![UC - 1-H5-PXI1Slot2-ai1](https://user-images.githubusercontent.com/46482002/169823967-06182415-9f8d-4558-b2c7-31b777ff0115.png)

### a2 - AE Unfiltered

The following is the time series plot of the data.

![UC - 1-H5-PXI1Slot2-ai2](https://user-images.githubusercontent.com/46482002/169811540-23648370-e020-42e3-ae8b-56326336fe0d.png)

The following is the wavelet plot for the first second of the recording. This just the response of the noise floor. The magnitude of the response is very small peaking a bit above 0.35. The band of dynamics is approximately between the scales 10 and 20.

![UC - 1-PXI1Slot2-ai2-wavelet-tt-0-1](https://user-images.githubusercontent.com/46482002/169811882-de9c19de-8ef0-423a-b61b-65cf575bda7a.png)

After about 1.2 seconds, the magnitude of the response increases drastically to around 3.5 at max. The most active dynamics are within the same 10-20 scale band with a secondary band appearing around 40-60 on the scale axis.

![UC - 1-PXI1Slot2-ai2-wavelet-tt-2-3](https://user-images.githubusercontent.com/46482002/169813241-94ada553-ee36-4f4b-b7ff-74c3e4f34d6e.png)

The magnitude increases to 8.5 and begins to quieten down until 10.5s. After that point the magnitude decreases back down to between 0.35-0.45. The activity between 10-20 scale remains and it appears the secondary dynamic band dissapears.

![UC - 1-PXI1Slot2-ai2-wavelet-tt-10-11](https://user-images.githubusercontent.com/46482002/169814227-0861b123-be5e-4904-bd55-2831a0c07204.png)

### a3 - AE RMS Filtered

The AE signal from a2 is filtered by applying the RMS filter.

![UC - 1-H5-PXI1Slot2-ai3](https://user-images.githubusercontent.com/46482002/169814289-4d5108f3-710f-4869-9b5d-e57cd02d4118.png)

The wavelet dynamics aren't visible across the recording. The magnitude during the most active section increases to 2.0.

### a4 - Vibration X

The signal regularly appears to cap at -10 to +10 suggesting a limit on the sensor.

![UC - 1-H5-PXI1Slot2-ai4](https://user-images.githubusercontent.com/46482002/169815238-50a7ddf5-e279-4309-8570-6360969773c6.png)

The signal starts off with three interesting bands of dynamics. The most active dynamics at the start is between scales 4 and 8. The 2nd set of dynamics is between 20 and 40. The 3rd one is rather unusual and is between 50 and 80. The cause of the speckling is unknown at the moment. The magnitude peaks around 5-5.5.

![UC - 1-PXI1Slot2-ai4-wavelet-tt-0-1](https://user-images.githubusercontent.com/46482002/169815510-5ffb9a26-9650-48c0-8b4e-9e8ca8f1394c.png)

When the tool is cutting, the magnitude increases to 30-35 and the 3 bands of activity become stringere. The 2nd band betweeen 20 and 40 sppears to be the strongest.

![UC - 1-PXI1Slot2-ai4-wavelet-tt-3-4](https://user-images.githubusercontent.com/46482002/169815573-5c578ba5-26c6-4f9a-b784-2b9314f4a9dc.png)

The dynamics cut off and suddenly drop after approximately 12.7 seconds with the activity between 20 and 40 drastically reducing in response. The overall magnitude of the response also decreases over time decreasing to 0.06-0.07 peak at the end.

![UC - 1-PXI1Slot2-ai2-wavelet-tt-12-13](https://user-images.githubusercontent.com/46482002/169816307-6c5f94e8-e208-443b-8537-f41877fc4eb7.png)

### a5 - Vibration Z

The signal shares similar dynamics to Vibration X including regularly capping at -10 to +10 suggesting a limit on the sensor.

![UC - 1-H5-PXI1Slot2-ai5](https://user-images.githubusercontent.com/46482002/169816470-4d32b9bb-764b-418a-976e-ab076b9e96f6.png)

There are two bands of activity at the start. The first between scales 4 and 8. The speckling dynamic is present in distinct lines at 20, 30, 60, 70 and 80. The magnitude peaks at around 5-6.

![UC - 1-PXI1Slot2-ai5-wavelet-tt-0-1](https://user-images.githubusercontent.com/46482002/169817508-7ba7e353-c430-46f3-9a7a-eee6d0220d41.png)

When the tool starts cutting the peak magnitude increases to approx 40. The activity between scales 4-8 and 40-80 increases in magnitude rapidly.

![UC - 1-PXI1Slot2-ai5-wavelet-tt-1-2](https://user-images.githubusercontent.com/46482002/169817892-cfe2a73b-fc34-4f5e-b3e4-b13d4e3459de.png)

After 10.6s the magnitude drops drastically to 16 and then to 5.

![UC - 1-PXI1Slot2-ai5-wavelet-tt-10-11](https://user-images.githubusercontent.com/46482002/169818238-6eec437e-cee1-4c7b-9d71-fa694187816d.png)

After 12.7s the band of activity between 40-80 drops in magnitude.

![UC - 1-PXI1Slot2-ai5-wavelet-tt-12-13](https://user-images.githubusercontent.com/46482002/169818449-d117f12c-8d91-42d7-b65a-ca12a55624d3.png)

### a6 - Feed Drive Current

The signal has distinct blocks of activty. It has a distinct noise floor that can be seen at the start and end with an approx magnitude of 2.45 - 2.55 volts. The period when the tool is retracting between 10 and 14 seconds is a distinct with the min/max magnitude staying more firmly between limits. The sharp discontinuities are suspected to be caused by the firmware too.

![UC - 1-H5-PXI1Slot2-ai6](https://user-images.githubusercontent.com/46482002/169818534-5107aee4-5a3a-4faf-8a9b-cfee9361b066.png)

The colormapping is either faulty or there is a single pixel causing the contrast to be severely reduced. Identifying changes in the dynamics is rather tricky to spot. 

When the tool starts cutting, some bands of activity appear. One appears between scales 40-80 and the dynamics between 4 and 6 increase in magnitude. The maximum magnitude doesn't change much at 5-5.5.

![UC - 1-PXI1Slot2-ai5-wavelet-tt-19-20](https://user-images.githubusercontent.com/46482002/169820292-5e228c4e-ffca-45ac-a11c-857034c7646e.png)

At around 8.8 seconds, these bands appear to decrease in magnitude but then re-appear again around 10s.

![UC - 1-PXI1Slot2-ai6-wavelet-tt-8-9](https://user-images.githubusercontent.com/46482002/169820358-c09db6c3-ac3f-42cb-a822-074ae1507c22.png)

After 10s, the dynamics in the higher scales (30+) is a lot fuzzier and spreads over a wider range. These wider dynamics stop just after 12.7s.

![UC - 1-PXI1Slot2-ai6-wavelet-tt-10-11](https://user-images.githubusercontent.com/46482002/169820707-524bc329-7333-49bc-a3af-7f72959b2672.png)

![UC - 1-PXI1Slot2-ai6-wavelet-tt-12-13](https://user-images.githubusercontent.com/46482002/169820921-5bbecaec-b299-4b3a-9228-0e23ebedb810.png)

### a7 - Spindle Drive

The signal has very erratic dynamics across the surface of the recording. It has a noise floor between 2.5 and 2.6 Volts that is present at the bedining and end. When the tool starts cutting around 1-1.2s, signal increases drastically to around 3.5-4 volts. At around 3 seconds, the voltage ramps up to around 4-4.2 volts at approximately 6 seconds. There is a block of activity after that which remains fairly consistently between 0 and 5v for around 1.5s. This is suspected to be when it's cutting into Aluminium.

![UC - 1-H5-PXI1Slot2-ai7](https://user-images.githubusercontent.com/46482002/169821102-901357b8-df01-4a89-8e7f-dce220ad8c0c.png)

The wavelet at the start is quiet with no distinctive activity. After 1.1 secs, certain dynamics become more noticeable. One set appears between 4 and 6. The 2nd set is more widespread covering the majority of the scale range. The max magnitude peaks close to 5.

![UC - 1-PXI1Slot2-ai7-wavelet-tt-1-2](https://user-images.githubusercontent.com/46482002/169822398-f898581a-b764-4b8a-a6be-4fc09f6e97a1.png)

After some time, the most distinct dynamics in that 2nd range appear to be between 40 and 60. There is some activity at the upper limits of the scale range (100) suggesting that it might stretch further. Between 6 and 7 secodns the max response peaks around 11.

![UC - 1-PXI1Slot2-ai7-wavelet-tt-5-6](https://user-images.githubusercontent.com/46482002/169822977-7552dcf5-117e-40ab-86a6-9c5d1e6593d3.png)

After approx 10.4 seconds, the dynamics begin to disappear and the max magnitude decreases to around 5.

![UC - 1-PXI1Slot2-ai7-wavelet-tt-10-11](https://user-images.githubusercontent.com/46482002/169823431-a028b025-55cd-42e9-b5c9-3fe1a66110da.png)

## Filtering

### Root-Mean-Square (RMS) Filtering

The RMS is performed using the following line of code. An approach based on Pandas is used as the data from the TDMS channel is read in as a Pandas DataFrame.
```python data_rms = (pd.DataFrame(np.abs(data.values.flatten())**2).rolling(N).mean())**0.5 ```

### Discrete Wavelet Transform (DWT)

### Continuous Wavelet Transform (CWT)

The CWT produces a 2D array of values. Each row is the response at a specific scale across the entire time period. Each column is the response across the scale range at each time sample. The DWT denoising approach thresholds each detail level by calculating the threshold from detail features. Taking a similar approach, the CWT components are thresholded to a value calculated from features. Two sources of the threshold were trialled. The first is from the global features of the entire response array and the second is thresholding at each scale separately.

#### Threshold from Global

#### Threshold from each Scale Level

### Fourier Frequency Filtering

Filtering at specific frequencies is a classic approach where the response across a specific range of frequencies is set to 0.0. The frequencies targeted are from the dynamics identified in the wavelet response for each signal.

## Principal Component Analysis (PCA)

Principle Component Analysis (PCA) transforms the data into a representative set of vectors known as components. One of the prime uses of PCA is to reduce the dimensionality of features, which are often several dimensions, into an easier to handle array. For example, the CWT response array tends to be very large as it covered the entire time period of the signal.

Before performing PCA, the data is standardised so that the mean is at 0.

![image](https://user-images.githubusercontent.com/46482002/170294721-d2b700e9-7e02-429b-a699-12c749948a62.png)
[source](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)

PCA reveals to what extent the features are correlated to each other. To emperically measure this, the covariance matrix between variables is calculated and attention is directed towards the sign of the covariance. If +ve then the two variables decrease and increase with eachother showing that they're correlated. If -ve then the two variables are inversely correlated.

Choosing the number of components is important as one of the requirements of the representation is to contain the same information as the original data. The components reresents the changes in the data that cause the maximal amount of variance. With a target of 95% variance, the number of components tended to be 8. It was also 10+ on occasion.

### Classifier

Given such a rich dataset both in length and sensor types, it's reasonable to use the signals in a classifier a property such as which material the drill is doing through.

The features passed through are based on the wavelet response of the classifier. First the given data segment, either a portion or the entire signl, is processed using the continuous wavelet transform (CWT) set to scales 1 to 100. This is fixed regardless of segment length so that the feature size is always the same. This is transposed to form a matrix where the feature columns are the different scale responses and each row is a sample. To reduce this to a more manageable feature set, PCA is applied to reduce it. Based on previous testing, the number of components is set to 8 which reduces the matrix to a 99 x 8 matrix of features.

#### Identifying Materials

One application of the classifier is identifying what material is being drilled. The labels for this are set based on the changepoints MAT files created earlier representing the segments of the files. Each segment is a distinct period of activity within the signal (e.g. drilling through material no. 1, transitioning into material no. 2 etc.) so it good for clearly defining the different material states. For simplicity and scaleability, the labels are organised as follows.

1 : Air
2: Transition
3 : Material no. 1
4 : Material no. 2
...
N : Material no. M
N+1 : Retraction.

This first attempt is flawed due to the imbalance of labels. Half of the signal is classified as retraction whilst there are fewer instances of materials and transitions. The resulting classifier is not going to be good but provides a way to test viability and also sets up the backend so it can be adjusted later.

## Denoising

Due to the high sampling rate, the signals are very noisy and obscures the signal. This affects the ability to fit breakpoints to the signal so needs to be denoised or at the very least have the noise reduced to improve the odds of a decent fitting.

### Root Mean Square (RMS)

RMS is an agressive form of filters based on a rolling window. A rolling window is a fixed size subset of a signal where the difference between the start of one window and the next is one index.

![](https://www.mathworks.com/help/econ/rollingwindow.png)

In RMS, the values in the window are replaced with the square root of the mean of the square of the signal. This heavily smooths the signal and removes a lot of spectral information but, with the correct window size, maintains the overall dynamics of the period.

| N=1000 | N=10000 |N=100,000|
|--------|---------|---------|
|![](![UC - 1-H1-PXI1Slot2-ai0-rms-smooth-N-1000](https://user-images.githubusercontent.com/46482002/173356929-1fde4fb2-2796-48d8-a478-594595b7dd32.png)| ![UC - 1-H1-PXI1Slot2-ai0-rms-smooth-N-10000](https://user-images.githubusercontent.com/46482002/173356954-f6acd5eb-014b-4bfd-b662-364d687e1257.png)| ![UC - 1-H1-PXI1Slot2-ai0-rms-smooth-N-100000](https://user-images.githubusercontent.com/46482002/173356979-b3f43aea-e739-461a-bfc8-e0c97a7e07da.png)|

### Discrete Wavelet Transform (DWT)

DWT represents a signal as a sum of an approximate signal and a set of details. The number of details or levels is set by the user and is capped by the mother wavelet being used. Denoising a signal under this representation involves attenuating certain levels of detail representing the noise.

The detail levels are attentuated based on statistical features of a target level. The threshold is calculated as the mean of the absolute difference between the signal and the mean of the signal. The result is then scaled by a magic factor which most seem to set to 1/0.6745. Not sure the source of it, but most use this value. Each detail level is then set to this results times the square root of two times the length.

$MAD =  \overline{||d_l-\overline{d_l}||}$

$\sigma = (1.0/0.6745)* MAD$

$\mu = \sigma * \sqrt{2*\ln{N}}$

Then every level of detail is then thresholded $\mu$ so that any values greater than it are set to it.

![UC - 1-H1-PXI1Slot2-ai0-dwt-filter-db6-levels-9](https://user-images.githubusercontent.com/46482002/173357416-b47ef701-df8c-4905-b8fd-940fa34cea46.png)

Unlike RMS, this maintains the spectral information to a certain degree just with certain details quieter than others.

### Continuous Wavelet Transform (CWT)

CWT is like hitting the signal with different tuning forks and recording the response. A stronger response means that the fork's components are strongly present in the signal. The base fork's design is set by the mother wavelet and each fork is a stretched/compressed version of it testing for different types of responses. To be more technical, the CWT tests for the presence of scaled responses of the sampling frequency.

The threshold is based on the statistical features of the different scale responses or the entire response set. For some reason, the filtered response for both approaches were very similar so only one plot is being shown.

![UC - 6-H5-PXI1Slot2-ai0-cwt-denoised](https://user-images.githubusercontent.com/46482002/173374818-828725f5-8cb8-424b-972e-a18d3d2cb9b9.png)

This approach wasn't as successful due to the nature of CWTs. Attenuating a certain scale response could reduce a component that was keeping a drastic change in values in check. In all denoised results, the amplitude is several times higher (x10) than the original and there are several artefacts such as spikes at the start and end of the signal. It also mirrors the reponse but given that we know the signals never go below 0 we can set the negative values to 0.

### Spectral Gating

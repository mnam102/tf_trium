import random
import pywt
import scaleogram as scg 
from skimage.transform import resize
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# 그래프 시각화 옵션 


def sample_to_scaleogram(df, signal_length=64, wavelet='mexh', cmap='gray'):
    scales =scg.periods2scales( np.arange(1, signal_length+1) ) # range of scales 
    # random한 idx 고르기
    idx = random.choice(range(len(df)))
    
    coeffs, freqs= pywt.cwt(df.iloc[idx,4:], scales, wavelet=wavelet) 
    rescale_coeffs = resize(coeffs, (signal_length, signal_length), mode = 'constant')
    
    x_tick_size = df.iloc[: ,4:].shape[1]
    
    plt.figure(figsize=(12, 5))
    plt.plot(df.iloc[idx,4:].T.reset_index(drop=True))
    plt.xticks(np.arange(0, x_tick_size, 200))
    plt.legend(['Raw data'])
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Absorbance')
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.imshow(coeffs, cmap = cmap, aspect = 'auto')
    plt.title('Original rectangualar shape')
    plt.xticks(np.arange(0, x_tick_size, 200))
    plt.xlabel('Time: Wavelength [nm]')
    plt.ylabel('Scale')
    plt.show()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(rescale_coeffs, cmap = cmap, aspect = 'auto')
    plt.title('Resized squared shape')
    plt.xlabel('Down sampled time: Wavelength [nm]')
    plt.ylabel('Scale')
    plt.show()
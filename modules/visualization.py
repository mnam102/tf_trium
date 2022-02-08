import random
import pywt
import scaleogram as scg 
from skimage.transform import resize
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
# 그래프 시각화 옵션 
from modules.metrics import MAPE
from sklearn.metrics import mean_absolute_error, r2_score


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
    
def get_scatter_result(y1_ans_f1, y1_pred_f1, target='F1', save_path='Hydrocarbon_dataset_result'):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1,1,1)
    # Plot results
    plt.scatter(y1_ans_f1, y1_pred_f1)
    ax.plot([0, 70000], [0, 70000], "--k")
    ax.set_ylabel("Target predicted")
    ax.set_xlabel("True Target")
    ax.set_title(target)
    ax.text(
        4500,
        70500,
        r"$R^2$=%.2f, MAE=%.2f, MAPE=%.4f"
        % (r2_score(y1_ans_f1, y1_pred_f1), mean_absolute_error(y1_ans_f1, y1_pred_f1)
           ,MAPE(y1_ans_f1, y1_pred_f1)/100),
    )
    ax.set_xlim([0, 80000])
    ax.set_ylim([0, 80000])

    fig.suptitle("Hydrocarbon Dataset", y=0.035)
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    
    plt.savefig(f'results/{target}_{save_path}.png')


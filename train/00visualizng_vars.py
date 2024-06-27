import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy import fft
from scipy.stats import gmean
import tqdm
from sklearn.model_selection import train_test_split,cross_val_score
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from seaborn import pairplot
import numpy.linalg as LA
import scipy.stats as sp

#change file and you will get plot of different csv
file= 'train/1720.csv'
data= pd.read_csv(file)
data[['b1x', 'b1y', 'b4x', 'b4y']] = data['b1x;b1y;b4x;b4y'].str.split(';', expand=True)
data = data.drop(columns=['b1x;b1y;b4x;b4y'])
data[['b1x', 'b1y', 'b4x', 'b4y']] = data[['b1x', 'b1y', 'b4x', 'b4y']].apply(pd.to_numeric)

fig, axes = plt.subplots(3, 2, figsize=(12, 12))

fft_x=abs(fft.fft(data['b4x'].to_numpy()))
fft_y=abs(fft.fft(data['b4y'].to_numpy()))
fft_acce=np.array([fft_y,fft_x])[:,1:]
fft_acce_norm=LA.norm(fft_acce,axis=0)
x = np.linspace(0, fft_x[1:].max(), 1000)

fitted_shape, loc, fitted_scale = sp.gamma.fit(fft_x[1:])
# Calculate 95% confidence interval
upper_bound_x = sp.gamma.ppf(0.95, fitted_shape, loc=loc, scale=fitted_scale)
fitted_shape, loc, fitted_scale = sp.gamma.fit(fft_y[1:])
upper_bound_y = sp.gamma.ppf(0.95, fitted_shape, loc=loc, scale=fitted_scale)

# Find points outside the 95% confidence interval
Color_cond=((fft_x[1:] > upper_bound_x) & (fft_y[1:] > upper_bound_y))
axes[0,0].scatter(fft_x[1:],fft_y[1:],s=0.5,c=np.array(['blue', 'red'])[(Color_cond).astype(int)])
axes[0,0].set_xlabel('FFT-x Amplitude')
axes[0,0].set_ylabel('FFT-y Amplitude')
axes[0,0].set_title('Scatter Plot Of Outlier Detection\n On FFT Data Spread Using 0.95 Confidence Interval')

pdf_expon = sp.expon.pdf(x, scale=fft_x[1:].mean())
pdf_gamma = sp.gamma.pdf(x, fitted_shape, loc=loc, scale=fitted_scale)

axes[1,0].hist(fft_x[1:], bins=30, density=True, alpha=0.6, color='g', label='Data Histogram')
axes[1,0].plot(x,pdf_expon)
axes[1,0].plot(x,pdf_gamma)
axes[1,0].legend(['Histogram Spread Of FFT-x value','Exponential Distribution On Data','Gamma Distribution On Data'])
axes[1,0].set_xlabel('FFT-x Magnitude')
axes[1,0].set_ylabel('Density')
axes[1,0].set_title('Fitted Distributions On FFT-x Data ')

fitted_shape, loc, fitted_scale = sp.gamma.fit(fft_acce_norm)
upper_bound = sp.gamma.ppf(1- 1e-5, fitted_shape, loc=loc, scale=fitted_scale)

Color_cond=(fft_acce_norm > upper_bound) 
axes[0,1].scatter(fft_x[1:],fft_y[1:],s=0.5,c=np.array(['blue', 'red'])[(Color_cond).astype(int)])
axes[0,1].set_xlabel('FFT-x Amplitude')
axes[0,1].set_ylabel('FFT-y Amplitude')
axes[0,1].set_title('Scatter Plot Of Outlier Detection\n On FFT Data Spread Using Normalized FFT & (1 - 1e-10) Confidence Interval')

axes[1,1].hist(fft_acce_norm,bins=30, density=True, alpha=0.6, color='g', label='Data Histogram')
axes[1,1].plot(x,sp.gamma.pdf(x, fitted_shape, loc=loc, scale=fitted_scale))
axes[1,1].legend(['Histogram Spread Of Normalized FFT value','Gamma Distribution On Data'])
axes[1,1].set_xlabel('Normalized FFT Magnitude')
axes[1,1].set_ylabel('Density')
axes[1,1].set_title('Fitted Distributions On Normalized FFT Data ')

fitted_shape, loc, fitted_scale = sp.gamma.fit(fft_x[1:])

Q1 = np.percentile(fft_acce_norm,25,axis=0)
Q3 = np.percentile(fft_acce_norm,75,axis=0)

IQR = Q3 - Q1

upper_bound = Q3 + 1.5 * IQR
Color_cond=(fft_acce_norm>upper_bound) 

axes[2,1].scatter(fft_x[1:],fft_y[1:],s=0.5,c=np.array(['blue', 'red'])[(Color_cond).astype(int)])
axes[2,1].set_xlabel('FFT-x Amplitude')
axes[2,1].set_ylabel('FFT-y Amplitude')
axes[2,1].set_title('Scatter Plot Of Outlier Detection\n On FFT Data Spread Using Normalized FFT & IQR')

Q1 = np.percentile(fft_acce,25,axis=1)
Q3 = np.percentile(fft_acce,75,axis=1)

IQR = Q3 - Q1

upper_bound = Q3 + 1.5 * IQR

Color_cond=(fft_acce[0]>upper_bound[0]) & (fft_acce[1]>upper_bound[1])

axes[2,0].scatter(fft_x[1:],fft_y[1:],s=0.5,c=np.array(['blue', 'red'])[(Color_cond).astype(int)])
axes[2,0].set_xlabel('FFT-x Amplitude')
axes[2,0].set_ylabel('FFT-y Amplitude')
axes[2,0].set_title('Scatter Plot Of Outlier Detection\n On FFT Data Spread Using IQR')

legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Non-outlier'),
                   plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5, label='Outlier')]

axes[0, 0].legend(handles=legend_elements, loc='lower right')
axes[0, 1].legend(handles=legend_elements, loc='lower right')
axes[2, 0].legend(handles=legend_elements, loc='lower right')
axes[2, 1].legend(handles=legend_elements, loc='lower right')
fig.suptitle(f'Scatter Plots of Outlier Detection Methods For {file}\n\n',fontsize=18,fontweight='bold')

plt.tight_layout()
plt.show()
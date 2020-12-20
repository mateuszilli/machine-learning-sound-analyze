# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import statistics
import scipy.fftpack as fftpk
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as py
import plotly.graph_objs as go
from scipy.io import wavfile
from scipy.fft import fft
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from google.colab import drive

drive.mount('/content/drive')
path = '/content/drive/samples/'

def extract_signal_attributes(FFT_signal):
  min_signal = min(FFT_signal)
  max_signal = max(FFT_signal)
  mean_signal = FFT_signal.mean()
  power_signal = np.power(FFT_signal, 2)
  power_mean_signal = power_signal.mean()
  energy_signal = np.sum(np.abs(power_signal))
  variance_signal = statistics.variance(FFT_signal)
  rms_signal = np.sqrt(power_mean_signal)
  shape_factor_signal = rms_signal / mean_signal
  crest_factor_signal = max_signal / rms_signal

  return [min_signal, max_signal, mean_signal, power_mean_signal, energy_signal, variance_signal, rms_signal, shape_factor_signal, crest_factor_signal]

signals = pd.DataFrame(columns=['min_signal', 'max_signal', 'mean_signal', 'power_mean_signal', 'energy_signal', 'variance_signal', 'rms_signal', 'shape_factor_signal', 'crest_factor_signal'])

for x in range(1, 140):
  s_rate, signal_bad = wavfile.read(path + 'bad/' + str(x) + '.wav')
  FFT_signal_bad = abs(fft(signal_bad))
  signal_attributes = extract_signal_attributes(FFT_signal_bad)
  signals = signals.append(
      {
        'min_signal': signal_attributes[0], 
        'max_signal': signal_attributes[1], 
        'mean_signal': signal_attributes[2], 
        'power_mean_signal': signal_attributes[3], 
        'energy_signal': signal_attributes[4], 
        'variance_signal': signal_attributes[5], 
        'rms_signal': signal_attributes[6], 
        'shape_factor_signal': signal_attributes[7], 
        'crest_factor_signal': signal_attributes[8], 
        'label': 0
      },
      ignore_index=True
)

for x in range(1, 140):
  s_rate, signal_good = wavfile.read(path + 'good/' + str(x) + '.wav')
  FFT_signal_good = abs(fft(signal_good))
  signal_attributes = extract_signal_attributes(FFT_signal_good)
  signals = signals.append(
      {
        'min_signal': signal_attributes[0], 
        'max_signal': signal_attributes[1], 
        'mean_signal': signal_attributes[2], 
        'power_mean_signal': signal_attributes[3], 
        'energy_signal': signal_attributes[4], 
        'variance_signal': signal_attributes[5], 
        'rms_signal': signal_attributes[6], 
        'shape_factor_signal': signal_attributes[7], 
        'crest_factor_signal': signal_attributes[8], 
        'label': 1
      },
      ignore_index=True
)

signals.head()

signals.describe()

signals.info()

s_rate, signal_bad = wavfile.read(path + 'bad/1.wav')
s_rate, signal_good = wavfile.read(path + 'good/1.wav')

trace_bad = go.Scatter(
    x = np.arange(0, len(signal_bad), 1), 
    y = signal_bad,
    mode = 'lines',
    name = 'Sound Signal Bad'
)

trace_good = go.Scatter(
    x = np.arange(0, len(signal_good), 1), 
    y = signal_good,
    mode = 'lines',
    name = 'Sound Signal Good'
)

data = [trace_bad, trace_good]

layout = go.Layout(
    title = 'Sound Graphic',
    xaxis = {'title': 'Y'},
    yaxis = {'title': 'X'}
)

figure = go.Figure(data = data, layout = layout)

figure

FFT_signal_bad = abs(fft(signal_bad))
f_signal_bad = fftpk.fftfreq(len(FFT_signal_bad), (1 / s_rate))

FFT_signal_good = abs(fft(signal_good))
f_signal_good = fftpk.fftfreq(len(FFT_signal_good), (1 / s_rate))

trace_bad = go.Scatter(
    x = f_signal_bad[range(len(FFT_signal_bad)//2)], 
    y = FFT_signal_bad[range(len(FFT_signal_bad)//2)],
    mode = 'lines',
    name = 'Sound Signal Bad'
)

trace_good = go.Scatter(
    x = f_signal_good[range(len(FFT_signal_good)//2)], 
    y = FFT_signal_good[range(len(FFT_signal_good)//2)],
    mode = 'lines',
    name = 'Sound Signal Good'
)

data = [trace_bad, trace_good]

layout = go.Layout(
    title = 'Sound Graphic',
    xaxis = {'title': 'Frequency'},
    yaxis = {'title': 'Amplitude'}
)

figure = go.Figure(data = data, layout = layout)

figure

correlation = signals.corr(method='pearson')
correlation.style.background_gradient(cmap='coolwarm')

cols = ['min_signal', 'max_signal', 'mean_signal','power_mean_signal', 'energy_signal', 'variance_signal', 'rms_signal', 'shape_factor_signal', 'crest_factor_signal']
signals[cols] = signals[cols].apply(minmax_scale)

x = signals[['mean_signal', 'power_mean_signal', 'energy_signal', 'variance_signal', 'rms_signal']]

y = signals['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

pipeline = Pipeline([('classifier', SVC())])

search_space = [
  {'classifier': [KNeighborsClassifier()], 'classifier__n_neighbors': [1, 5, 10, 50]},
  {'classifier': [DecisionTreeClassifier(random_state = 1)], 'classifier__criterion': ['gini', 'entropy']},
  {'classifier': [SVC(random_state = 1)], 'classifier__C': [1, 5, 10, 50]}
]

base_search_CV = GridSearchCV(pipeline, search_space, cv=3)

base_search_CV_fit = base_search_CV.fit(x_train, y_train)

best_model = base_search_CV_fit.best_estimator_

best_model.fit(x_train, y_train)

best_model_predict = best_model.predict(x_test)

accuracy_score(y_test, best_model_predict)

recall_score(y_test, best_model_predict)

precision_score(y_test, best_model_predict)

f1_score(y_test, best_model_predict)

confusion_matrix(y_test, best_model_predict)

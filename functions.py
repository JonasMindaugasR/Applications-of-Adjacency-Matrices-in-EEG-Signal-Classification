import scipy.io
import pandas as pd
import plotly.graph_objects as go
import os
import random
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import scipy.stats as stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


import torch
from torch.nn import Linear
import torch.nn.functional as F
import numpy as np
import pickle

from scipy import signal
from scipy.signal import butter, filtfilt

import numpy as np
from xgboost import XGBClassifier
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils import resample
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

import mne
import tqdm
import concurrent.futures

from collections import Counter

#--------------------Importing--------------------

def edf_to_numpy(edf_file):
  '''
  Converts file of .edf format to numpy array
  :param edf_file:
  :return: data (numpy array, channels X time interval), channel names (list)
  '''
  # Load .edf file
  raw = mne.io.read_raw_edf(edf_file, preload=True)

  # Get data and channel names
  data = raw.get_data()
  channel_names = raw.ch_names

  return data, channel_names


# Eyes open
def eye_opened_conn_metric(sub_id, dataset_root, fs, int_end, notch_filt, l_cut, h_cut):
  # Loading .edf file
  subj_prefix = f"S{sub_id + 1:03}"
  subj_dir = f"{dataset_root}/{subj_prefix}"
  baseline_eyes_open = f"{subj_dir}/{subj_prefix}R01.edf"
  num_array, ch_names = edf_to_numpy(baseline_eyes_open)

  # Preprocessing
  info = mne.create_info(ch_names, fs)
  loadedRaw = mne.io.RawArray(num_array[:, :int_end], info)

  # Remove frequencies
  loadedRaw.notch_filter([notch_filt], picks=ch_names)
  loadedRaw.filter(l_freq=l_cut, h_freq=h_cut, picks=ch_names)  # Only keeping frequencies between 1-50 Hz

  # Downsampling the data
  loadedRaw.resample(120, npad='auto')
  result = loadedRaw.get_data()

  return {'result': result, 'label': 1}


# Eyes closed
def eye_closed_conn_metric(sub_id, dataset_root, fs, int_end, notch_filt, l_cut, h_cut):
  subj_prefix = f"S{sub_id + 1:03}"
  subj_dir = f"{dataset_root}/{subj_prefix}"
  baseline_eyes_closed = f"{subj_dir}/{subj_prefix}R02.edf"

  num_array, ch_names = edf_to_numpy(baseline_eyes_closed)

  # Preprocessing
  info = mne.create_info(ch_names, fs)
  loadedRaw = mne.io.RawArray(num_array[:, :int_end], info)

  # Remove frequencies
  loadedRaw.notch_filter([notch_filt], picks=ch_names)
  loadedRaw.filter(l_freq=l_cut, h_freq=h_cut, picks=ch_names)  # Only keeping frequencies between 1-50 Hz

  # Downsampling the data
  loadedRaw.resample(120, npad='auto')
  result = loadedRaw.get_data()

  return {'result': result, 'label': 0}


def import_using_multi_threading(dataset_root, store_dir, fs, int_end, notch_filt, l_cut, h_cut):
  eeg_array = []
  with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = []
    for k in tqdm.tqdm(range(0, 100, 10)):
      for i in range(k, k + 10):
        futures.append(executor.submit(eye_opened_conn_metric, i, dataset_root, fs, int_end, notch_filt, l_cut, h_cut))
        futures.append(executor.submit(eye_closed_conn_metric, i, dataset_root, fs, int_end, notch_filt, l_cut, h_cut))
    for future in concurrent.futures.as_completed(futures):
      eeg_array.append(future.result())

  np.save(f'{store_dir}/eeg_eyes_numpy_arrays_{l_cut}-{h_cut}.npy', eeg_array)

def contr_conn_metric(path, fs, int_end, notch_filt, l_cut, h_cut):
  num_array, ch_names = edf_to_numpy(path)

  # preprocessing
  # create mne raw file
  info = mne.create_info(ch_names, fs)
  loadedRaw = mne.io.RawArray(num_array[:, :int_end], info)

  loadedRaw.notch_filter([notch_filt], picks=ch_names)
  loadedRaw.filter(l_freq=l_cut, h_freq=h_cut, picks=ch_names)  # only keeping frequencies between 1-50 Hz
  # downsampling the data
  loadedRaw.resample(120, npad='auto')
  result = loadedRaw._data

  # np.save(f'{store_dir}/eeg_eyes_opened_raw_{sub_id}.npy', num_array)
  return {'result': result, 'label': 0}


# eyes open
def depr_conn_metric(path, fs, int_end, notch_filt, l_cut, h_cut):
  num_array, ch_names = edf_to_numpy(path)

  # preprocessing
  info = mne.create_info(ch_names, fs)
  loadedRaw = mne.io.RawArray(num_array[:, :int_end], info)

  # remove frequencies
  loadedRaw.notch_filter([notch_filt], picks=ch_names)
  loadedRaw.filter(l_freq=0.5, h_freq=4.0, picks=ch_names)  # only keeping frequencies between 1-50 Hz

  # downsampling the data
  loadedRaw.resample(120, npad='auto')
  result = loadedRaw._data

  return {'result': result, 'label': 1}


def import_depr_using_multi_threading(folder_path_healthy, folder_path_depr, store_dir, fs, int_end, notch_filt, l_cut, h_cut):
  eeg_array = []

  file_paths_healthy = [os.path.join(folder_path_healthy, f) for f in os.listdir(folder_path_healthy) if
                        os.path.isfile(os.path.join(folder_path_healthy, f))]

  file_paths_depr = [os.path.join(folder_path_depr, f) for f in os.listdir(folder_path_depr) if
                     os.path.isfile(os.path.join(folder_path_depr, f))]

  with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    for k in tqdm.tqdm(range(0, 100, 10)):
      for i in range(k, k + 10):
        control = file_paths_healthy[i]
        future = executor.submit(contr_conn_metric, control, fs, int_end, notch_filt, l_cut, h_cut)
        eeg_array.append(future.result())

        depr = file_paths_depr[i]
        future_2 = executor.submit(depr_conn_metric, depr, fs, int_end, notch_filt, l_cut, h_cut)
        eeg_array.append(future_2.result())

  np.save(f'{store_dir}/depression_numpy_arrays_{l_cut}-{h_cut}.npy', eeg_array)


#--------------------Utility--------------------

# converting matrices to normalization
def _normalize_trial(trial):
  trial_avg = np.mean(trial)
  trial_std = np.std(trial)

  trial = (trial - trial_avg)/trial_std
  return trial

# def min_max(trial):
#   trial_min  = np.min(trial)
#   trial_max  = np.max(trial)

#   trial = (trial - trial_min)/(trial_max - trial_min)
#   return trial

def binarize(trial, threshold = 0):
  if trial > threshold:
    trial = 1
  else:
    trial = 0
  return trial

def _normalize_trial_eeg(eeg_trial, eeg_mean, eeg_std):
  eeg_trial_normalized = (eeg_trial - eeg_mean) / eeg_std

  return eeg_trial_normalized

#--------------------Pearson's correlation--------------------

# using correlation
def corr_features(trial, binarize, threshold, num_electrodes):
  feat = []
  trial_df = pd.DataFrame(trial, columns=list(range(1, num_electrodes+1)))
  corr_matrix = np.array(trial_df.corr())

  corr_matrix = _normalize_trial(corr_matrix)

  if binarize == True:
    corr_matrix = np.where(corr_matrix > threshold, 1, 0)

  for i in range(np.shape(corr_matrix)[0]):
    feat.append(list(np.squeeze(corr_matrix[i, :])))

  return feat

#--------------------Phase locking value--------------------

# phase locking value -  PLV values represent the degree of synchronization (phase locking) between different pairs of channels in the trial
def phase_locking_value(theta1, theta2):
    complex_phase_diff = np.exp(complex(0,1)*(theta1 - theta2))
    plv = np.abs(np.sum(complex_phase_diff))/len(theta1)
    return plv

# phase locking value adjancency matrix
def plv_corr_matrix(trial, num_channels):
  corr_matrix = np.zeros((num_channels,num_channels))
  for i in range(num_channels):
    for j in range(num_channels):
      corr_matrix[i, j] = phase_locking_value(trial[:, i], trial[:, j])
      corr_matrix[j, i] = phase_locking_value(trial[:, i], trial[:, j])

  return(corr_matrix)

# extracting features from plv adjancency matrix - extracts features, which correspond to rows in adjancency matrix and shows plv of one electrode to others
def plv_features(trial, binarize=False, threshold = 0):
  feat = []
  num_channels = np.shape(trial)[1]
  corr_matrix = _normalize_trial(plv_corr_matrix(trial, num_channels))

  if binarize == True:
    corr_matrix = np.where(corr_matrix > threshold, 1, 0)

  for i in range(np.shape(corr_matrix)[0]):
    feat.append(list(np.squeeze(corr_matrix[i, :])))
  return feat

#--------------------Phase lag index--------------------

# phase lag  index
def phase_lag_index(theta1, theta2):
    complex_phase_diff = np.sin(np.sign(theta1 - theta2))
    pli = np.abs(np.sum(complex_phase_diff))/len(theta1)
    return pli

# phase locking index adjancency matrix
def pli_corr_matrix(trial, num_channels):
  corr_matrix = np.zeros((num_channels,num_channels))
  for i in range(num_channels):
    for j in range(num_channels):
      corr_matrix[i, j] = phase_lag_index(trial[:, i], trial[:, j])
      corr_matrix[j, i] = phase_lag_index(trial[:, i], trial[:, j])

  return(corr_matrix)

# extracting features from pli adjancency matrix - extracts features, which correspond to rows in adjancency matrix and shows pli of one electrode to others
def pli_features(trial, binarize=False, threshold = 0):
  feat = []
  num_channels = np.shape(trial)[1]
  corr_matrix = _normalize_trial(pli_corr_matrix(trial, num_channels))

  if binarize == True:
    corr_matrix = np.where(corr_matrix > threshold, 1, 0)

  for i in range(np.shape(corr_matrix)[0]):
    feat.append(list(np.squeeze(corr_matrix[i, :])))
  return feat

#--------------------Imaginary part of coherence--------------------

# imaginary part of coherence
def imag_part_coherence(theta1, theta2, fs = 120):
    # Calculate the cross-spectral density (CSD)
    _, Pxy = signal.csd(theta1, theta2, fs=fs)

    # Extract the imaginary part of the CSD
    imag_Pxy = np.imag(Pxy)
    coh = np.mean(imag_Pxy)
    return coh

# imaginary part of coherence adjancency matrix
def imag_part_coherence_matrix(trial, num_channels):
  imag_part_coh_matrix = np.zeros((num_channels,num_channels))
  for i in range(num_channels):
    for j in range(num_channels):
      imag_part_coh_matrix[i, j] = imag_part_coherence(trial[:, i], trial[:, j])
      imag_part_coh_matrix[j, i] = imag_part_coherence(trial[:, i], trial[:, j])

  return(imag_part_coh_matrix)

# extracting features from imaginary part of coherence adjancency matrix
def imag_part_coh_features(trial, binarize=False, threshold = 0):
  feat = []
  num_channels = np.shape(trial)[1]
  imag_part_coh_matrix = _normalize_trial(imag_part_coherence_matrix(trial, num_channels))

  if binarize == True:
    imag_part_coh_matrix = np.where(imag_part_coh_matrix > threshold, 1, 0)

  for i in range(np.shape(imag_part_coh_matrix)[0]):
    feat.append(list(np.squeeze(imag_part_coh_matrix[i, :])))
  return feat

#--------------------Data preparation--------------------

# returns list of subjects for closed and opened
def data_preparation(path, int_start, int_end, normalize):
  # load data
  eeg_data = np.load(path, allow_pickle=True)

  # Initialize an empty dictionary
  eeg_data_new = {}

  # Loop through the list of dictionaries
  for pair in eeg_data:
      # Extract the key and value from each dictionary
      for key, value in pair.items():
          # If the key is not in the dictionary, add it with an empty list
          if key not in eeg_data_new:
              eeg_data_new[key] = []

          # Append the value to the list associated with the key
          eeg_data_new[key].append(value)

  # Print the result
  eeg_data_signals = np.stack(eeg_data_new['result'], axis = 0)

  # taking shorter interval and permuting data
  eeg_data_permuted = np.transpose(eeg_data_signals[:,:,int_start:int_end], (0, 2, 1))
  eeg_label_data = np.array(eeg_data_new['label'])

  opened = list()
  closed = list()

  if normalize==True:

    eeg_mean = np.mean(eeg_data_permuted)
    eeg_std = np.std(eeg_data_permuted)

    for i, k in enumerate(eeg_label_data):
      if k == 1:
        opened.append(_normalize_trial_eeg(eeg_data_permuted[i,:,:], eeg_mean, eeg_std))
      else:
        closed.append(_normalize_trial_eeg(eeg_data_permuted[i,:,:], eeg_mean, eeg_std))

  if normalize==False:
    for i, k in enumerate(eeg_label_data):
      if k == 1:
        opened.append(eeg_data_permuted[i,:,:]) #do not normalize trial for data_eyes
      else:
        closed.append(eeg_data_permuted[i,:,:])

  return opened, closed

#--------------------Filtering--------------------

# Butterworth bandpass filter design
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Apply bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)  # Filtering along the time axis
    return y

# apply filter to dataset
def filter_dataset(list_mat1, list_mat2, lowcut, highcut, fs, order=4):
  opened = []
  closed = []

  for i in list_mat1:
    # Filter the signal to keep frequencies between 1 and 50 Hz
    filtered_signal = bandpass_filter(i, lowcut, highcut, fs, order)

    opened.append(filtered_signal)


  for i in list_mat2:
    # Filter the signal to keep frequencies between 1 and 50 Hz
    filtered_signal = bandpass_filter(i, lowcut, highcut, fs, order)

    closed.append(filtered_signal)

  return opened, closed

#--------------------Utility 2--------------------

# apply adjacency metric
def calc_adj_features(opened, closed, adj_fun, binarize=False, threshold=0, num_electrodes=21):
  opened_corr = list()
  closed_corr = list()

  if adj_fun == corr_features:
    for i in range(len(closed)):
      opened_corr.append(adj_fun(opened[i], binarize=binarize, threshold=threshold, num_electrodes=num_electrodes))
      closed_corr.append(adj_fun(closed[i], binarize=binarize, threshold=threshold, num_electrodes=num_electrodes))
  else:
    for i in range(len(closed)):
      opened_corr.append(adj_fun(opened[i], binarize=binarize, threshold=threshold))
      closed_corr.append(adj_fun(closed[i], binarize=binarize, threshold=threshold))

  # convert to numpy array
  opened_corr = np.stack(opened_corr)
  closed_corr = np.stack(closed_corr)

  return opened_corr, closed_corr


# flatten adjancency (and extract only upper triangle)
def flatten_adj_mat(opened_corr, closed_corr, upp_triangle=False, num_electrodes=64):
  opened_corr_flat = list()
  closed_corr_flat = list()

  for i in range(len(opened_corr)):
    if upp_triangle == False:
      opened_corr_flat.append(opened_corr[i].flatten())
      closed_corr_flat.append(closed_corr[i].flatten())

    elif upp_triangle == True:
      opened_corr_flat.append(opened_corr[i][np.triu_indices(num_electrodes, k=1)])

      closed_corr_flat.append(opened_corr[i][np.triu_indices(num_electrodes, k=1)])

  return opened_corr_flat, closed_corr_flat

#--------------------ML functions--------------------

def prep_ml_dataset(features_mat_1, features_mat_2, metric=False):
  if metric == False:
    opened_corr_flat = list()
    closed_corr_flat = list()

    for i in range(len(features_mat_1)):
      opened_corr_flat.append(features_mat_1[i].flatten())
      closed_corr_flat.append(features_mat_2[i].flatten())

    X = opened_corr_flat + closed_corr_flat
    y = [1] * 100 + [0] * 100  # Labels

  if metric == True:
    X = features_mat_1 + features_mat_2
    y = [1] * 100 + [0] * 100  # Labels

  # Split the data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Standardize the data
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  return X_train_scaled, X_test_scaled, y_train, y_test

def lasso_optimization(X_train_scaled, y_train, alpha=0.1):
  selected_features = []
  while not selected_features:
    # Apply Lasso for feature selection
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_scaled, y_train)

    # Display the selected features
    selected_features = [i for i, coef in enumerate(lasso.coef_) if coef != 0]

    if not selected_features:
      alpha *= 0.5  # Decrease alpha (you can adjust the reduction factor)

  return selected_features

def svm_lasso_bootsrap(X_train_scaled, X_test_scaled, y_train, y_test, selected_features, n_bootstrap=100):
  # Create a reduced feature matrix with only the selected features
  X_train_selected = X_train_scaled[:, selected_features]
  X_test_selected = X_test_scaled[:, selected_features]

  accuracies = []
  f1_scores = []

  # Step 2: Bootstrapping process on the selected features
  for _ in range(n_bootstrap):
    # Bootstrap resampling
    X_train_bootstrap, y_train_bootstrap = resample(X_train_selected, y_train)

    # Train a classifier (SVM) using only the selected features
    svm = SVC(kernel="rbf", C=1.0)
    svm.fit(X_train_bootstrap, y_train_bootstrap)

    # Make predictions and evaluate the classifier
    y_pred = svm.predict(X_test_selected)

    # Evaluate the model
    accuracies.append(accuracy_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

  method = "svm"
  mean_accuracy = np.mean(accuracies) * 100
  mean_f1_score = np.mean(f1_scores) * 100
  ci_accuracy = [np.percentile(accuracies, 2.5) * 100, np.percentile(accuracies, 97.5) * 100]
  ci_f1_score = [np.percentile(f1_scores, 2.5) * 100, np.percentile(f1_scores, 97.5) * 100]

  return method, mean_accuracy, mean_f1_score, ci_accuracy, ci_f1_score


def rf_lasso_bootsrap(X_train_scaled, X_test_scaled, y_train, y_test, selected_features, n_bootstrap=100, n_estimators=100, max_depth=None):
  # Create a reduced feature matrix with only the selected features
  X_train_selected = X_train_scaled[:, selected_features]
  X_test_selected = X_test_scaled[:, selected_features]

  accuracies = []
  f1_scores = []

  # Step 2: Bootstrapping process on the selected features
  for _ in range(n_bootstrap):
    # Bootstrap resampling
    X_train_bootstrap, y_train_bootstrap = resample(X_train_selected, y_train)

    # Create a Random Forest model
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    # Train the Random Forest model on the resampled training data
    rf_model.fit(X_train_bootstrap, y_train_bootstrap)

    # Predict labels for the test data
    y_pred = rf_model.predict(X_test_selected)

    # Evaluate the model
    accuracies.append(accuracy_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

  method = "rf"
  mean_accuracy = np.mean(accuracies) * 100
  mean_f1_score = np.mean(f1_scores) * 100
  ci_accuracy = [np.percentile(accuracies, 2.5) * 100, np.percentile(accuracies, 97.5) * 100]
  ci_f1_score = [np.percentile(f1_scores, 2.5) * 100, np.percentile(f1_scores, 97.5) * 100]

  return method, mean_accuracy, mean_f1_score, ci_accuracy, ci_f1_score


def xgb_lasso_bootsrap(X_train_scaled, X_test_scaled, y_train, y_test, selected_features, n_bootstrap=100, n_estimators=200, max_depth=3, learning_rate=0.3):
  # Create a reduced feature matrix with only the selected features
  X_train_selected = X_train_scaled[:, selected_features]
  X_test_selected = X_test_scaled[:, selected_features]

  accuracies = []
  f1_scores = []

  # Step 2: Bootstrapping process on the selected features
  for _ in range(n_bootstrap):
    # Bootstrap resampling
    X_train_bootstrap, y_train_bootstrap = resample(X_train_selected, y_train)

    # Create an XGBoost model
    xgb_model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, eval_metric='logloss')

    # Train the XGBoost model on the resampled training data
    xgb_model.fit(X_train_bootstrap, y_train_bootstrap)

    # Predict labels for the test data
    y_pred = xgb_model.predict(X_test_selected)

    # Evaluate the model
    accuracies.append(accuracy_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

  method = "xgb"
  mean_accuracy = np.mean(accuracies) * 100
  mean_f1_score = np.mean(f1_scores) * 100
  ci_accuracy = [np.percentile(accuracies, 2.5) * 100, np.percentile(accuracies, 97.5) * 100]
  ci_f1_score = [np.percentile(f1_scores, 2.5) * 100, np.percentile(f1_scores, 97.5) * 100]

  return method, mean_accuracy, mean_f1_score, ci_accuracy, ci_f1_score
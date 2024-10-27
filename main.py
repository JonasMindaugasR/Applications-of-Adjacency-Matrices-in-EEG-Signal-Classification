####### Directories
path_eyes = '/content/drive/MyDrive/master workflow/eeg_eyes_numpy_arrays_2.npy'
path_depr_all = '/content/drive/MyDrive/master workflow/depression_numpy_arrays.npy'
path_delta = '/content/drive/MyDrive/master workflow/depression_numpy_arrays_0.5-4.npy'
path_theta = '/content/drive/MyDrive/master workflow/depression_numpy_arrays_4-7.npy'
path_alpha = '/content/drive/MyDrive/master workflow/depression_numpy_arrays_8-12.npy'
path_sigma = '/content/drive/MyDrive/master workflow/depression_numpy_arrays_12-16.npy'
path_beta = '/content/drive/MyDrive/master workflow/depression_numpy_arrays_13-30.npy'

path = [path_eyes]
waves = [path_depr_all, path_delta, path_theta, path_alpha, path_sigma, path_beta]
lowcut = [1.0, 0.5, 4.0, 8.0, 12.0, 13.0]
highcut = [50.0, 4.0, 7.0, 12.0, 16.0, 30.0]

num_electrodes = 21

# signal interval
int_start = 7000
int_end = 7500

# Example signal with 500 time points and 64 electrodes
fs = 120  # Sampling frequency (Hz)

# Bandpass filter settings (1-50 Hz)
lowcut = 12.0  # Lower bound of the frequency band (1 Hz)
highcut = 16.0  # Upper bound of the frequency band (50 Hz)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

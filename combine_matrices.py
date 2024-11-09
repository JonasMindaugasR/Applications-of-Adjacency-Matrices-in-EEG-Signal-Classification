import functions as f
import pandas as pd
import tqdm
import os
import re
import multiprocessing as mp
import numpy as np

dataset = "depression"
num_electrodes = 20
truncate_electrodes = True

fs = 256
int_start = 2000
int_end = 2500

# dataset = "eyes"
# num_electrodes = 64
# truncate_electrodes = False
#
# fs = 160
# int_start = 4000
# int_end = 4500

input_dir = f"H:/magistro_studijos/magis/data_{dataset}/graph_output"
output_dir = f"H:/magistro_studijos/magis/data_{dataset}/graph_combined_output"

def combine_tensors(tensor1, tensor2):
    num_of_subjects = tensor1.shape[0]
    tensor1_size = tensor1.shape[1]
    tensor2_size = tensor2.shape[1]

    combined_tensor_size = tensor1_size + tensor2_size

    # Initialize the larger tensor of shape (100, 30, 30)
    combined_tensor = np.zeros((num_of_subjects, combined_tensor_size, combined_tensor_size))

    # Insert tensor1 in the top-left corner
    combined_tensor[:, :tensor1_size, :tensor1_size] = tensor1

    # Insert tensor2 in the bottom-right corner
    combined_tensor[:, tensor1_size:, tensor1_size:] = tensor2

    return combined_tensor

def combine_bands(filename_0):

    filename_1 = filename_0.replace("label_0", "label_1")

    pattern = r"(plv_features|pli_features|corr_features|imag_part_coh_features).*_(bin|weight)_0.5-4.0"

    match = re.search(pattern, filename_0)
    if match:
        feature, label = match.groups()

    band_1_label_1 = np.load(f"{input_dir}/{feature}_label_1_{label}_0.5-4.0.npy")
    band_2_label_1 = np.load(f"{input_dir}/{feature}_label_1_{label}_4.0-8.0.npy")
    band_3_label_1 = np.load(f"{input_dir}/{feature}_label_1_{label}_8.0-12.0.npy")
    band_4_label_1 = np.load(f"{input_dir}/{feature}_label_1_{label}_12.0-30.0.npy")

    combined_tensors_label_1 = combine_tensors(band_1_label_1, band_2_label_1)
    combined_tensors_label_1 = combine_tensors(combined_tensors_label_1, band_3_label_1)
    combined_tensors_label_1 = combine_tensors(combined_tensors_label_1, band_4_label_1)



    band_1_label_0 = np.load(f"{input_dir}/{feature}_label_0_{label}_0.5-4.0.npy")
    band_2_label_0 = np.load(f"{input_dir}/{feature}_label_0_{label}_4.0-8.0.npy")
    band_3_label_0 = np.load(f"{input_dir}/{feature}_label_0_{label}_8.0-12.0.npy")
    band_4_label_0 = np.load(f"{input_dir}/{feature}_label_0_{label}_12.0-30.0.npy")

    combined_tensors_label_0 = combine_tensors(band_1_label_0, band_2_label_0)
    combined_tensors_label_0 = combine_tensors(combined_tensors_label_0, band_3_label_0)
    combined_tensors_label_0 = combine_tensors(combined_tensors_label_0, band_4_label_0)

    np.save(f"{output_dir}/{feature}_label_1_{label}_combined", combined_tensors_label_1)
    np.save(f"{output_dir}/{feature}_label_0_{label}_combined", combined_tensors_label_0)

    return
def combine_metrics(filename_0):
    filename_1 = filename_0.replace("label_0", "label_1")

    pattern = r"(plv_features|pli_features|corr_features|imag_part_coh_features).*_(bin|weight)_(\d+\.\d+-\d+\.\d+)"

    match = re.search(pattern, filename_0)
    if match:
        _, label, freq = match.groups()

    band_1_label_1 = np.load(f"{input_dir}/plv_features_label_1_{label}_{freq}.npy")
    band_2_label_1 = np.load(f"{input_dir}/pli_features_label_1_{label}_{freq}.npy")
    band_3_label_1 = np.load(f"{input_dir}/corr_features_label_1_{label}_{freq}.npy")
    band_4_label_1 = np.load(f"{input_dir}/imag_part_coh_features_label_1_{label}_{freq}.npy")

    combined_tensors_label_1 = combine_tensors(band_1_label_1, band_2_label_1)
    combined_tensors_label_1 = combine_tensors(combined_tensors_label_1, band_3_label_1)
    combined_tensors_label_1 = combine_tensors(combined_tensors_label_1, band_4_label_1)



    band_1_label_0 = np.load(f"{input_dir}/plv_features_label_0_{label}_{freq}.npy")
    band_2_label_0 = np.load(f"{input_dir}/pli_features_label_0_{label}_{freq}.npy")
    band_3_label_0 = np.load(f"{input_dir}/corr_features_label_0_{label}_{freq}.npy")
    band_4_label_0 = np.load(f"{input_dir}/imag_part_coh_features_label_0_{label}_{freq}.npy")

    combined_tensors_label_0 = combine_tensors(band_1_label_0, band_2_label_0)
    combined_tensors_label_0 = combine_tensors(combined_tensors_label_0, band_3_label_0)
    combined_tensors_label_0 = combine_tensors(combined_tensors_label_0, band_4_label_0)

    np.save(f"{output_dir}/metrics_{freq}_label_1_{label}_combined", combined_tensors_label_1)
    np.save(f"{output_dir}/metrics_{freq}_label_0_{label}_combined", combined_tensors_label_0)

    return

if __name__ == '__main__':
    os.makedirs(f"{output_dir}", exist_ok=True)
    # Get list of filenames
    filenames = [f for f in os.listdir(input_dir) if 'label_0' in f and '0.5-4.0' in f]

    # Use multiprocessing Pool to parallelize file processing
    with mp.Pool(mp.cpu_count() - 3) as pool:
        # Collect all results in a list of lists
        list(tqdm.tqdm(pool.imap(combine_bands, filenames), total=len(filenames)))

    filenames = [f for f in os.listdir(input_dir) if 'label_0' in f and 'corr_features' in f]

    # Use multiprocessing Pool to parallelize file processing
    with mp.Pool(mp.cpu_count() - 3) as pool:
        # Collect all results in a list of lists
        list(tqdm.tqdm(pool.imap(combine_metrics, filenames), total=len(filenames)))

import functions as f
import numpy as np
from pathlib import Path
import tqdm
import os
import re
import multiprocessing as mp

dataset = "depression"
num_electrodes = 20
truncate_electrodes = True

fs = 256
int_start = 1000
int_end = 5000

# dataset = "eyes"
# num_electrodes = 64
# truncate_electrodes = False
#
# fs = 160
# int_start = 4000
# int_end = 4500

input_dir = f"H:/magistro_studijos/magis/data_{dataset}/output"
output_dir = f"H:/magistro_studijos/magis/data_{dataset}/graph_output"

def calc_graph(filename):

    match = re.search(r'(\d+\.?\d*)-(\d+\.?\d*)', filename)

    if match:
        label_1, label_0 = f.data_preparation(path=f"{input_dir}/{filename}",
                                              int_start=int_start,
                                              int_end=int_end,
                                              normalize="electrode",
                                              truncate_electrodes=truncate_electrodes, filter=True,
                                              lowcut=float(match[1]), highcut=float(match[2]), fs=fs)

        for binarize in [True, False]:
            for feature in [f.plv_features, f.corr_features, f.pli_features, f.imag_part_coh_features]:
                label_1_adj, label_0_adj = f.calc_adj_features(label_1, label_0,
                                                               feature, binarize=binarize, threshold=0,
                                                               num_electrodes=num_electrodes)

                if binarize == True:
                    np.save(f"{output_dir}/{feature.__name__}_label_1_bin_{match[0]}", label_1_adj)
                    np.save(f"{output_dir}/{feature.__name__}_label_0_bin_{match[0]}", label_0_adj)
                elif binarize == False:
                    np.save(f"{output_dir}/{feature.__name__}_label_1_weight_{match[0]}", label_1_adj)
                    np.save(f"{output_dir}/{feature.__name__}_label_0_weight_{match[0]}", label_0_adj)



if __name__ == '__main__':
    # Get list of filenames
    os.makedirs(f"{output_dir}", exist_ok=True)
    filenames = os.listdir(input_dir)

    # Use multiprocessing Pool to parallelize file processing
    with mp.Pool(mp.cpu_count()-3) as pool:
        # Collect all results in a list of lists
        list(tqdm.tqdm(pool.imap(calc_graph, filenames), total=len(filenames)))
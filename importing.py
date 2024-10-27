import functions as f
import time

dataset = "depression" # "eyes"

dataset_root = r"H:/magistro_studijos/magis/kodai/dyconnmap-master/dyconnmap-master/examples/data/raw_data/eeg-motor-movementimagery-dataset-1.0.0/files"
store_dir_eyes = r"H:/magistro_studijos/magis/data_eyes/output"


folder_path_healthy = r"H:/magistro_studijos/magis/data_depression/raw/edf nevalyti atskirti/sveiki"
folder_path_depr = r"H:/magistro_studijos/magis/data_depression/raw/edf nevalyti atskirti/depresija"
store_dir_depr = r"H:/magistro_studijos/magis/data_depression/output"

fs = 120
int_end = 9000
notch_filt = 50
l_cut = 1.0
h_cut = 40.0


if __name__ == '__main__':
    start_time_multi = time.time()
    if dataset == "eyes":
        f.import_using_multi_threading(dataset_root=dataset_root,
                                       store_dir=store_dir_eyes,
                                       fs=fs,
                                       int_end=int_end,
                                       notch_filt=notch_filt,
                                       l_cut=l_cut,
                                       h_cut=h_cut)
    elif dataset == "depression":
        f.import_depr_using_multi_threading(folder_path_healthy=folder_path_healthy,
                                            folder_path_depr=folder_path_depr,
                                            store_dir=store_dir_depr,
                                            fs=fs,
                                            int_end=int_end,
                                            notch_filt=notch_filt,
                                            l_cut=l_cut,
                                            h_cut=h_cut)
    end_time_multi = time.time()
    print(end_time_multi-start_time_multi)
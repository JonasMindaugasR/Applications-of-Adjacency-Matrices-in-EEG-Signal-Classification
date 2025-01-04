import functions as f
import time

dataset = "eyes" # "eyes" "depression"

dataset_root = r"H:/magistro_studijos/magis/kodai/dyconnmap-master/dyconnmap-master/examples/data/raw_data/eeg-motor-movementimagery-dataset-1.0.0/files"
store_dir_eyes = r"H:/magistro_studijos/magis/final_results/data_eyes/output"


folder_path_healthy = r"H:/magistro_studijos/magis/data_depression/raw/edf nevalyti atskirti/sveiki"
folder_path_depr = r"H:/magistro_studijos/magis/data_depression/raw/edf nevalyti atskirti/depresija"
store_dir_depr = r"H:/magistro_studijos/magis/final_results/data_depression/output"


fs_eyes = 160
fs_depression = 256
int_end_eyes = 9000
int_end_depr = 5000
notch_filt = 50
lowcut = [0.5, 0.5, 4.0, 8.0, 12.0, 30.0]
highcut = [40.0, 4.0, 8.0, 12.0, 30.0, 40.0]


if __name__ == '__main__':
    start_time_multi = time.time()
    for i in range(6):
        if dataset == "eyes":
            f.import_using_multi_threading(dataset_root=dataset_root,
                                           store_dir=store_dir_eyes,
                                           fs=fs_eyes,
                                           int_end=int_end_eyes,
                                           notch_filt=notch_filt,
                                           l_cut=lowcut[i],
                                           h_cut=highcut[i])
        elif dataset == "depression":
            f.import_depr_using_multi_threading(folder_path_healthy=folder_path_healthy,
                                                folder_path_depr=folder_path_depr,
                                                store_dir=store_dir_depr,
                                                fs=fs_depression,
                                                int_end=int_end_depr,
                                                notch_filt=notch_filt,
                                                l_cut=lowcut[i],
                                                h_cut=highcut[i])

    # if dataset == "depression":
    #     f.import_depr_using_multi_threading_no_filt(folder_path_healthy=folder_path_healthy,
    #                                                 folder_path_depr=folder_path_depr,
    #                                                 store_dir=store_dir_depr,
    #                                                 fs=fs_depression,
    #                                                 int_end=int_end_depr)
    end_time_multi = time.time()
    print(end_time_multi-start_time_multi)
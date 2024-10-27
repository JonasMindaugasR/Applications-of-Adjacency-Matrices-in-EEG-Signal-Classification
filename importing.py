import functions as f
import time

dataset_root = r"H:/magistro_studijos/magis/kodai/dyconnmap-master/dyconnmap-master/examples/data/raw_data/eeg-motor-movementimagery-dataset-1.0.0/files"
store_dir = r"H:/magistro_studijos/magis/data_eyes/output"
fs = 120
int_end = 9000
notch_filt = 50
l_cut = 1.0
h_cut = 40.0


if __name__ == '__main__':
    start_time_multi = time.time()
    f.import_using_multi_threading(dataset_root=dataset_root,
                                   store_dir=store_dir,
                                   fs=fs,
                                   int_end=int_end,
                                   notch_filt=notch_filt,
                                   l_cut=l_cut,
                                   h_cut=h_cut)
    end_time_multi = time.time()
    print(end_time_multi-start_time_multi)
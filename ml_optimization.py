import functions as f
import pandas as pd
import openpyxl
import tqdm
import os
import re
import multiprocessing as mp

# dataset = "depression"
# num_electrodes = 20
# truncate_electrodes = True
#
# fs = 256
# int_start = 2000
# int_end = 2500

dataset = "eyes"
num_electrodes = 64
truncate_electrodes = False

fs = 160
int_start = 4500
int_end = 5000

input_dir = f"H:/magistro_studijos/magis/data_{dataset}/output"
output_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_output"

# Define a function that performs the processing for each file
def process_file(filename):
    # Initialize results list to accumulate each row as a dictionary
    results = []

    match = re.search(r'(\d+\.?\d*)-(\d+\.?\d*)', filename)
    if match:
        for normalize in ["electrode", "dataset", "not"]:
            opened_raw, closed_raw = f.data_preparation(
                path=f"{input_dir}/{filename}",
                int_start=int_start,
                int_end=int_end,
                normalize=normalize,
                truncate_electrodes=truncate_electrodes
            )

            opened, closed = f.filter_dataset(
                opened_raw, closed_raw,
                float(match.group(1)), float(match.group(2)), fs
            )

            for binarize in [True, False]:
                for upp_triangle in [True, False]:
                    for feature in [f.plv_features, f.corr_features, f.pli_features, f.imag_part_coh_features]:
                        opened_corr, closed_corr = f.calc_adj_features(
                            opened, closed,
                            feature, binarize=binarize, threshold=0,
                            num_electrodes=num_electrodes
                        )

                        opened_corr_flat, closed_corr_flat = f.flatten_adj_mat(
                            opened_corr, closed_corr,
                            upp_triangle=upp_triangle,
                            num_electrodes=num_electrodes
                        )

                        X_train_scaled, X_test_scaled, y_train, y_test = f.prep_ml_dataset(
                            opened_corr_flat, closed_corr_flat,
                            metric=False
                        )

                        selected_features = f.lasso_optimization(X_train_scaled, y_train, alpha=0.1)

                        for model in [f.svm_lasso_bootsrap, f.rf_lasso_bootsrap, f.xgb_lasso_bootsrap]:
                            method, mean_accuracy, mean_f1_score, ci_accuracy, ci_f1_score = model(
                                X_train_scaled, X_test_scaled,
                                y_train, y_test,
                                selected_features
                            )

                            # Collect results in a dictionary
                            results.append({
                                'interval start': int_start,
                                'interval end': int_end,
                                'normalized trial': normalize,
                                'l_freq': float(match.group(1)),
                                'h_freq': float(match.group(2)),
                                'binarized': binarize,
                                'upper triangle': upp_triangle,
                                'adj_feature': feature.__name__,
                                'method': method,
                                'Mean Accuracy': mean_accuracy,
                                'Mean F1 Score': mean_f1_score,
                                '95% CI for Accuracy Low': ci_accuracy[0],
                                '95% CI for Accuracy High': ci_accuracy[1],
                                '95% CI for F1 Score Low': ci_f1_score[0],
                                '95% CI for F1 Score High': ci_f1_score[1],
                                'selected features': selected_features
                            })
    return results

if __name__ == '__main__':
    # Get list of filenames
    filenames = os.listdir(input_dir)

    # Use multiprocessing Pool to parallelize file processing
    with mp.Pool(mp.cpu_count()-3) as pool:
        # Collect all results in a list of lists
        all_results = list(tqdm.tqdm(pool.imap(process_file, filenames), total=len(filenames)))

    # Flatten list of lists into a single list of dictionaries
    all_results = [item for sublist in all_results for item in sublist]

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save DataFrame to Excel
    results_df.to_excel(f"{output_dir}/df_result_{dataset}_{int_start}-{int_end}_{num_electrodes}_elec_total_v2.xlsx", index=False)
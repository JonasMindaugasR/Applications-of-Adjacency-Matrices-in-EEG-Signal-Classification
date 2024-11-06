import functions as f
import pandas as pd
import tqdm
import os
import re
import multiprocessing as mp
import numpy as np

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
int_start = 4000
int_end = 4500

input_dir = f"H:/magistro_studijos/magis/data_{dataset}/graph_output"
# input_dir = f"H:/magistro_studijos/magis/data_{dataset}/graph_output_{int_start}_{int_end}"
output_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_output"


# Function to perform Lasso grid search to find unique feature sets
def get_unique_lasso_features(X_train_scaled, y_train):
    alpha_values = np.logspace(-2, 1, 10)  # Example alpha grid: 0.001 to 10
    unique_feature_sets = []

    for alpha in alpha_values:
        selected_features = f.lasso_optimization(X_train_scaled, y_train, alpha=alpha)

        # Check for unique feature sets
        if selected_features not in unique_feature_sets:
            unique_feature_sets.append(selected_features)

    return unique_feature_sets


# Function to run ML models on each unique feature set
def run_ml_models(X_train_scaled, X_test_scaled, y_train, y_test, feature_set):
    results = []

    # Run each model on the unique feature set
    for model in [f.svm_lasso_bootsrap, f.rf_lasso_bootsrap, f.xgb_lasso_bootsrap]:
        method, mean_accuracy, mean_f1_score, ci_accuracy, ci_f1_score = model(
            X_train_scaled, X_test_scaled,
            y_train, y_test,
            feature_set
        )

        # Append results
        results.append({
            'method': method,
            'selected features': feature_set,
            'Mean Accuracy': mean_accuracy,
            'Mean F1 Score': mean_f1_score,
            '95% CI for Accuracy Low': ci_accuracy[0],
            '95% CI for Accuracy High': ci_accuracy[1],
            '95% CI for F1 Score Low': ci_f1_score[0],
            '95% CI for F1 Score High': ci_f1_score[1]
        })
    return results


def opt_lasso_ml_file(label_1_graph, label_0_graph):
    # Flatten adjacency matrices
    label_1_graph_flat, label_0_graph_flat = f.flatten_adj_mat(
        label_1_graph, label_0_graph,
        upp_triangle=False,
        num_electrodes=num_electrodes
    )

    # Prepare training and test sets
    X_train_scaled, X_test_scaled, y_train, y_test = f.prep_ml_dataset(
        label_1_graph_flat, label_0_graph_flat,
        metric=False
    )

    # Get unique sets of selected features
    unique_feature_sets = get_unique_lasso_features(X_train_scaled, y_train)

    # Collect results for each unique feature set
    all_results = []
    for feature_set in unique_feature_sets:
        results = run_ml_models(X_train_scaled, X_test_scaled, y_train, y_test, feature_set)
        all_results.extend(results)

    return all_results

def process_ml(filename_0):
    filename_1 = filename_0.replace("label_0", "label_1")

    pattern = r"(plv_features|pli_features|corr_features|imag_part_coh_features).*_(bin|weight)_(\d+\.\d+-\d+\.\d+)"

    match = re.search(pattern, filename_0)
    if match:
        feature, label, frequency_range = match.groups()

    label_1_graph = np.load(f"{input_dir}/{filename_1}")
    label_0_graph = np.load(f"{input_dir}/{filename_0}")

    results = opt_lasso_ml_file(label_1_graph, label_0_graph)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save DataFrame to Excel
    results_df.to_excel(f"{output_dir}/{feature}_{label}_{frequency_range}.xlsx",index=False)


    return "done"

if __name__ == '__main__':
    # Get list of filenames
    filenames = [f for f in os.listdir(input_dir) if 'label_0' in f]

    # Use multiprocessing Pool to parallelize file processing
    with mp.Pool(mp.cpu_count()-3) as pool:
        # Collect all results in a list of lists
        list(tqdm.tqdm(pool.imap(process_ml, filenames), total=len(filenames)))
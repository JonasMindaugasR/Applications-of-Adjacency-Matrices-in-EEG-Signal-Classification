import functions as f
import pandas as pd
import tqdm
import os
import re
import multiprocessing as mp
import numpy as np
from sklearn.decomposition import PCA

dataset = "eyes"
num_electrodes = 64
truncate_electrodes = False

fs = 160
int_start = 3500
int_end = 7500

# dataset = "depression"
# num_electrodes = 20
# truncate_electrodes = True
#
# fs = 256
# int_start = 1000
# int_end = 5000

input_dir = f"H:/magistro_studijos/magis/final_results_3/data_{dataset}/graph_output"
input_combined_dir = f"H:/magistro_studijos/magis/final_results_3/data_{dataset}/graph_combined_output"

# matrices_combination = 'metrics' # 'non' 'bands' 'metrics'
matrices_combinations = ['all'] # 'non','bands', 'metrics', 'all'




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
def run_ml_models_pca(X_train_scaled, X_test_scaled, y_train, y_test, feature_set=None, n_component=None):
    results = []

    # Run each model on the unique feature set
    for model in [f.svm_lasso, f.rf_lasso, f.xgb_lasso]:
        method, mean_accuracy, mean_f1_score = model(
            X_train_scaled, X_test_scaled,
            y_train, y_test,
            feature_set
        )

        # Append results
        results.append({
            'method': method,
            'selected features': feature_set,
            'n comps': n_component,
            'Mean Accuracy': mean_accuracy,
            'Mean F1 Score': mean_f1_score
        })
    return results

def run_ml_models(X_train_scaled, X_test_scaled, y_train, y_test, feature_set=None, n_component=None):
    results = []

    # Run each model on the unique feature set
    for model in [f.svm_lasso_bootsrap, f.rf_lasso_bootsrap, f.xgb_lasso_bootsrap]:
        method, mean_accuracy, mean_f1_score, std_accuracy = model(
            X_train_scaled, X_test_scaled,
            y_train, y_test,
            feature_set
        )

        # Append results
        results.append({
            'method': method,
            'selected features': feature_set,
            'n comps': n_component,
            'Mean Accuracy': mean_accuracy,
            'Mean F1 Score': mean_f1_score,
            'Std accuracy': std_accuracy
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
    X_train_scaled, X_test_scaled, y_train, y_test, _, _ = f.prep_ml_dataset(
        label_1_graph_flat, label_0_graph_flat,
        metric=False
    )

    # Get unique sets of selected features
    unique_feature_sets = get_unique_lasso_features(X_train_scaled, y_train)

    # Collect results for each unique feature set
    all_results = []
    for feature_set in unique_feature_sets:
        results = run_ml_models(X_train_scaled, X_test_scaled, y_train, y_test, feature_set=feature_set)
        all_results.extend(results)

    return all_results

def opt_pca_ml_file(label_1_graph, label_0_graph):
    # Flatten adjacency matrices
    label_1_graph_flat, label_0_graph_flat = f.flatten_adj_mat(
        label_1_graph, label_0_graph,
        upp_triangle=False,
        num_electrodes=num_electrodes
    )

    # Prepare training and test sets
    X_train_scaled, X_test_scaled, y_train, y_test, _, _ = f.prep_ml_dataset(
        label_1_graph_flat, label_0_graph_flat,
        metric=False
    )

    # create grid for components
    n_components = list(range(1, 11))

    all_results = []
    # iterate over number of pca components
    for n_component in n_components:
        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=n_component)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        # Run machine learning models on PCA-transformed data
        results = run_ml_models_pca(X_train_pca, X_test_pca, y_train, y_test, n_component=n_component)

        all_results.extend(results)

    return all_results

def process_ml(filename_0, method = "pca"):
    output_dir = f"H:/magistro_studijos/magis/final_results_3/data_{dataset}/ml_output_{method}/"
    os.makedirs(f"{output_dir}", exist_ok=True)

    filename_1 = filename_0.replace("label_0", "label_1")

    pattern = r"(plv_features|pli_features|corr_features|imag_part_coh_features).*_(bin|weight)_(\d+\.\d+-\d+\.\d+)"

    match = re.search(pattern, filename_0)
    if match:
        feature, label, frequency_range = match.groups()

    label_1_graph = np.load(f"{input_dir}/{filename_1}")
    label_0_graph = np.load(f"{input_dir}/{filename_0}")


    if method == "lasso":
        results = opt_lasso_ml_file(label_1_graph, label_0_graph)
    if method == "pca":
        results = opt_pca_ml_file(label_1_graph, label_0_graph)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save DataFrame to Excel
    results_df.to_excel(f"{output_dir}/{feature}_{label}_{frequency_range}.xlsx",index=False)


    return "done"

def process_ml_combined_metrics(filename_0, method = "pca"):
    output_dir = f"H:/magistro_studijos/magis/final_results_3/data_{dataset}/ml_output_{method}_combined_metrics/"
    os.makedirs(f"{output_dir}", exist_ok=True)

    filename_1 = filename_0.replace("label_0", "label_1")

    pattern = r"metrics_(\d+\.\d+-\d+\.\d+).*_(bin|weight)_combined"

    match = re.search(pattern, filename_0)
    if match:
        freq, label = match.groups()

    label_1_graph = np.load(f"{input_combined_dir}/{filename_1}")
    label_0_graph = np.load(f"{input_combined_dir}/{filename_0}")


    if method == "lasso":
        results = opt_lasso_ml_file(label_1_graph, label_0_graph)
    if method == "pca":
        results = opt_pca_ml_file(label_1_graph, label_0_graph)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save DataFrame to Excel
    results_df.to_excel(f"{output_dir}/metrics_{freq}_{label}_combined.xlsx",index=False)

    return "done"

def process_ml_combined_bands(filename_0, method = "pca"):
    output_dir = f"H:/magistro_studijos/magis/final_results_3/data_{dataset}/ml_output_{method}_combined_bands/"
    os.makedirs(f"{output_dir}", exist_ok=True)

    filename_1 = filename_0.replace("label_0", "label_1")

    pattern = r"(plv_features|pli_features|corr_features|imag_part_coh_features).*_(bin|weight)_combined"

    match = re.search(pattern, filename_0)
    if match:
        feature, label = match.groups()

    label_1_graph = np.load(f"{input_combined_dir}/{filename_1}")
    label_0_graph = np.load(f"{input_combined_dir}/{filename_0}")


    if method == "lasso":
        results = opt_lasso_ml_file(label_1_graph, label_0_graph)
    if method == "pca":
        results = opt_pca_ml_file(label_1_graph, label_0_graph)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save DataFrame to Excel
    results_df.to_excel(f"{output_dir}/{feature}_{label}_combined.xlsx",index=False)

    return "done"

def process_ml_combined_all(filename_0, method = "pca"):
    output_dir = f"H:/magistro_studijos/magis/final_results_3/data_{dataset}/ml_output_{method}_combined_all/"
    os.makedirs(f"{output_dir}", exist_ok=True)

    filename_1 = filename_0.replace("label_0", "label_1")

    pattern = r"metrics_freq_.*_(bin|weight)_combined_all\.npy"

    match = re.search(pattern, filename_0)
    if match:
        label = match.groups()

    label_1_graph = np.load(f"{input_combined_dir}/{filename_1}")
    label_0_graph = np.load(f"{input_combined_dir}/{filename_0}")


    if method == "lasso":
        results = opt_lasso_ml_file(label_1_graph, label_0_graph)
    if method == "pca":
        results = opt_pca_ml_file(label_1_graph, label_0_graph)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save DataFrame to Excel
    results_df.to_excel(f"{output_dir}/metrics_freq_{label[0]}_combined_all.xlsx",index=False)

    return "done"

if __name__ == '__main__':
            for matrices_combination in matrices_combinations:
                if matrices_combination == 'non':
                    # Get list of filenames
                    filenames = [f for f in os.listdir(input_dir) if 'label_0' in f]

                    # Use multiprocessing Pool to parallelize file processing
                    with mp.Pool(mp.cpu_count()-3) as pool:
                        # Collect all results in a list of lists
                        list(tqdm.tqdm(pool.imap(process_ml, filenames), total=len(filenames)))

                if matrices_combination == 'bands':
                    # Get list of filenames
                    filenames = [f for f in os.listdir(input_combined_dir) if 'label_0' in f and 'metrics' not in f]

                    # Use multiprocessing Pool to parallelize file processing
                    with mp.Pool(mp.cpu_count() - 3) as pool:
                        # Collect all results in a list of lists
                        list(tqdm.tqdm(pool.imap(process_ml_combined_bands, filenames), total=len(filenames)))

                if matrices_combination == 'metrics':
                    # Get list of filenames
                    filenames = [f for f in os.listdir(input_combined_dir) if 'label_0' in f and 'metrics' in f]

                    # Use multiprocessing Pool to parallelize file processing
                    with mp.Pool(mp.cpu_count() - 3) as pool:
                        # Collect all results in a list of lists
                        list(tqdm.tqdm(pool.imap(process_ml_combined_metrics, filenames), total=len(filenames)))

                if matrices_combination == 'all':
                    # Get list of filenames
                    filenames = [f for f in os.listdir(input_combined_dir) if 'label_0' in f and 'metrics_freq' in f]

                    # Use multiprocessing Pool to parallelize file processing
                    with mp.Pool(mp.cpu_count() - 3) as pool:
                        # Collect all results in a list of lists
                        list(tqdm.tqdm(pool.imap(process_ml_combined_all, filenames), total=len(filenames)))
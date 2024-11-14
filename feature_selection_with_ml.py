import functions as f
import pandas as pd
import tqdm
import os
import re
import multiprocessing as mp
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.model_selection import GridSearchCV, train_test_split
from grakel import Graph, WeisfeilerLehman, SvmTheta, ShortestPath, GraphletSampling

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt


dataset = "depression"
num_electrodes = 20
truncate_electrodes = True

fs = 256
int_start = 1000
int_end = 5000

# dataset = "depression_filt"
# num_electrodes = 20
# truncate_electrodes = True
#
# fs = 256
# int_start = 2000
# int_end = 2500

# dataset = "eyes"
# num_electrodes = 64
# truncate_electrodes = False
#
# fs = 160
# int_start = 4000
# int_end = 4500

matrices_combination = 'bands'

input_dir = f"H:/magistro_studijos/magis/data_{dataset}/graph_output_4000_int"
input_combined_dir = f"H:/magistro_studijos/magis/data_{dataset}/graph_combined_output_4000_int"
# input_dir = f"H:/magistro_studijos/magis/data_{dataset}/graph_output_{int_start}_{int_end}"
# output_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_output_{int_start}_{int_end}"

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
def run_ml_models(X_train_scaled, X_test_scaled, y_train, y_test, feature_set=None):
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

def opt_pca_ml_file(label_1_graph, label_0_graph, n_components=10):
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

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)


    # Run machine learning models on PCA-transformed data
    results = run_ml_models(X_train_pca, X_test_pca, y_train, y_test)

    return results

# TODO: FIX somehow breaks after one iteration
def opt_spectral_ml_file(label_1_graph, label_0_graph, n_components=3):
    # Create an empty list to store the embeddings for each subject
    subject_embeddings_1 = []
    subject_embeddings_0 = []

    # Iterate over each subject
    for subject in range(label_1_graph.shape[0]):
        # Access the adjacency matrix (graph) for the current subject
        subject_graph_1 = label_1_graph[subject, :, :]
        subject_graph_0 = label_0_graph[subject, :, :]

        # Handle NaN values by replacing with 0
        subject_graph_1 = np.nan_to_num(subject_graph_1, nan=0)
        subject_graph_0 = np.nan_to_num(subject_graph_0, nan=0)

        # Log any remaining NaNs (sanity check)
        if np.isnan(subject_graph_1).any() or np.isnan(subject_graph_0).any():
            raise ValueError("NaN values detected after replacement in subject graph.")

        # Initialize the spectral embedding model
        spectral_1 = SpectralEmbedding(n_components=n_components, affinity="precomputed")
        spectral_0 = SpectralEmbedding(n_components=n_components, affinity="precomputed")

        # Apply Spectral Embedding (Dimensionality Reduction)
        subject_embedding_1 = spectral_1.fit_transform(subject_graph_1)
        subject_embedding_0 = spectral_0.fit_transform(subject_graph_0)

        # Store the embedding
        subject_embeddings_1.append(subject_embedding_1)
        subject_embeddings_0.append(subject_embedding_0)

    # Convert the list of subject embeddings to a numpy array
    subject_embeddings_1 = np.array(subject_embeddings_1)
    subject_embeddings_0 = np.array(subject_embeddings_0)

    X_train, X_test, y_train, y_test = f.prep_ml_dataset(subject_embeddings_1, subject_embeddings_0)

    # Run machine learning models on the Spectral-decomposed data
    results = run_ml_models(X_train, X_test, y_train, y_test)

    return results



# Convert adjacency matrix to grakel-compatible graph format
def adjacency_to_grakel_graph(adjacency_matrix):
    # Convert adjacency matrix to GraKeL graph format with default labels
    n = len(adjacency_matrix)  # Number of nodes
    edges = [(i, j) for i in range(n) for j in range(n) if adjacency_matrix[i][j] != 0]

    # Assign default labels to each node if labels are not provided
    labels = {i: 1 for i in range(n)}  # Example label: '1' for all nodes

    return (edges, labels)

def opt_lasso_graph_kernels(label_1_graph, label_0_graph):
    # Flatten adjacency matrices
    label_1_graph_flat, label_0_graph_flat = f.flatten_adj_mat(
        label_1_graph, label_0_graph,
        upp_triangle=False,
        num_electrodes=num_electrodes
    )

    # Convert to NumPy arrays
    label_1_graph_flat = np.asarray(label_1_graph_flat)
    label_0_graph_flat = np.asarray(label_0_graph_flat)

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
        # Example: Assuming the selected feature set has 400 features, and you want to reshape it into 20x20
        num_samples_1 = len(label_1_graph_flat)
        num_samples_0 = len(label_0_graph_flat)

        # Initialize a zero matrix for each sample
        label_1_graph_selected = np.zeros_like(label_1_graph_flat)
        label_0_graph_selected = np.zeros_like(label_0_graph_flat)

        # Set the selected features to their original values, others remain 0
        label_1_graph_selected[:, feature_set] = label_1_graph_flat[:, feature_set]
        label_0_graph_selected[:, feature_set] = label_0_graph_flat[:, feature_set]

        # # Ensure the number of selected features matches 400 (for 20x20 matrix)
        # assert label_1_graph_selected.shape[1] == label_1_graph_selected.shape[1], "Feature set should have 400 features for 20x20 reshaping"
        # assert label_0_graph_selected.shape[1] == label_0_graph_selected.shape[1], "Feature set should have 400 features for 20x20 reshaping"

        if label_1_graph_selected.shape[1] == 400:
            # Reshape the selected features into square matrices (20x20)
            vec_1 = label_1_graph_selected.reshape(num_samples_1, 20, 20)  # shape: (num_samples_train, 20, 20)
            vec_0 = label_0_graph_selected.reshape(num_samples_0, 20, 20)  # shape: (num_samples_test, 20, 20)

        if label_1_graph_selected.shape[1] == 4096:
            # Reshape the selected features into square matrices (20x20)
            vec_1 = label_1_graph_selected.reshape(num_samples_1, 64, 64)  # shape: (num_samples_train, 20, 20)
            vec_0 = label_0_graph_selected.reshape(num_samples_0, 64, 64)  # shape: (num_samples_test, 20, 20)


        # Convert adjacency matrices to GraKeL graph format
        graphs_1 = [adjacency_to_grakel_graph(adj) for adj in vec_1]
        graphs_0 = [adjacency_to_grakel_graph(adj) for adj in vec_0]

        # Combine and label data
        graphs = graphs_1 + graphs_0
        labels = [1] * len(graphs_1) + [0] * len(graphs_0)

        # # Create the Weisfeiler-Lehman kernel with SvmTheta as the base kernel
        # base_graph_kernel = (SvmTheta, {})  # Pass SvmTheta with an empty parameter dictionary
        # wl_kernel = WeisfeilerLehman(base_graph_kernel=base_graph_kernel, n_iter=5)
        #
        # # Fit and transform the graphs
        # K = wl_kernel.fit_transform(graphs)

        # Create the Shortest Path kernel (or any other kernel)
        sp_kernel = ShortestPath()  # You can change this to another kernel if needed
        K = sp_kernel.fit_transform(graphs)

        # # Plot kernel matrix heatmap
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(K, cmap='viridis', cbar=True)
        # plt.title("Kernel Matrix Heatmap")
        # plt.show()

        # Step 1: Perform the train-test split
        train_indices, test_indices = train_test_split(np.arange(K.shape[0]), test_size=0.2, random_state=42)

        # Step 2: Create the training and testing kernel matrices
        K_train = K[train_indices][:, train_indices]  # 140x140 kernel matrix
        K_test = K[test_indices][:, train_indices]  # 60x140 kernel matrix (to test on training subjects)
        y_train = np.array(labels)[train_indices]  # Corresponding labels for training
        y_test = np.array(labels)[test_indices]  # Corresponding labels for testing

        # Initialize SVM classifier with precomputed kernel
        # Initialize parameters
        n_bootstrap = 3  # Number of bootstrap iterations
        accuracies = []

        # Perform bootstrap sampling
        for _ in range(n_bootstrap):
            # Create a bootstrap sample (sampling with replacement)
            indices = np.random.choice(len(K_train), size=len(K_train), replace=True)
            X_train_resampled = K_train[indices]
            y_train_resampled = y_train[indices]

            # Train the SVM classifier with the precomputed kernel on the resampled data
            svm = SVC(kernel="precomputed")
            svm.fit(X_train_resampled, y_train_resampled)

            # Evaluate the model on the test set
            y_pred = svm.predict(K_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        # Calculate the mean and standard deviation of accuracies across all bootstrap iterations
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        print(f"Mean Test Accuracy: {mean_accuracy:.2f}")
        print(f"Standard Deviation of Accuracy: {std_accuracy:.2f}")

        return "done"



def process_ml(filename_0, method = "lasso"):
    output_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_output_{method}_4000_int/"
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
    if method == "spectral":
        opt_lasso_graph_kernels(label_1_graph, label_0_graph)
    opt_lasso_graph_kernels(label_1_graph, label_0_graph)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save DataFrame to Excel
    results_df.to_excel(f"{output_dir}/{feature}_{label}_{frequency_range}.xlsx",index=False)


    return "done"

def process_ml_no_filt(filename_0, method):
    output_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_output_{method}/"
    os.makedirs(f"{output_dir}", exist_ok=True)

    filename_1 = filename_0.replace("label_0", "label_1")

    pattern = r"(plv_features|pli_features|corr_features|imag_part_coh_features).*_(bin|weight)_no_filt"

    match = re.search(pattern, filename_0)
    if match:
        feature, label = match.groups()

    label_1_graph = np.load(f"{input_dir}/{filename_1}")
    label_0_graph = np.load(f"{input_dir}/{filename_0}")

    if method == "lasso":
        results = opt_lasso_ml_file(label_1_graph, label_0_graph)
    if method == "pca":
        results = opt_pca_ml_file(label_1_graph, label_0_graph)
    if method == "spectral":
        results = opt_spectral_ml_file(label_1_graph, label_0_graph)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save DataFrame to Excel
    results_df.to_excel(f"{output_dir}/{feature}_{label}_no_filt.xlsx",index=False)


    return "done"

def process_ml_combined_metrics(filename_0, method = "lasso"):
    output_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_output_{method}_combined_metrics_4000_int/"
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
    if method == "spectral":
        results = opt_spectral_ml_file(label_1_graph, label_0_graph)
    # opt_lasso_graph_kernels(label_1_graph, label_0_graph)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save DataFrame to Excel
    results_df.to_excel(f"{output_dir}/metrics_{freq}_{label}_combined.xlsx",index=False)

    return "done"

def process_ml_combined_bands(filename_0, method = "lasso"):
    output_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_output_{method}_combined_bands_4000_int/"
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
    if method == "spectral":
        results = opt_spectral_ml_file(label_1_graph, label_0_graph)
    # opt_lasso_graph_kernels(label_1_graph, label_0_graph)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save DataFrame to Excel
    results_df.to_excel(f"{output_dir}/{feature}_{label}_combined.xlsx",index=False)

    return "done"

if __name__ == '__main__':
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
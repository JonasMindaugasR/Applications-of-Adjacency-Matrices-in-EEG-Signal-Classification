import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.decomposition import PCA
import functions as f
import tqdm
import ast

dataset = "eyes" # "depression", "eyes"


graph_type = ""
# graph_type = "combined_metrics"
# graph_type = "combined_bands"

# # Define the directories
# input_ml_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_output_lasso_4000_int"
# input_dir = f"H:/magistro_studijos/magis/data_{dataset}/graph_output"
# output_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_optim_output_4000_int"





# Initialize models and hyperparameters
models = {
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(eval_metric='mlogloss')
}

param_grids = {
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    'Random Forest': {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': [None, 'sqrt']
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'gamma': [0, 0.1]
    }
}

def get_metadata(dataset, graph_type, feat_select):
    if dataset == "eyes":
        if graph_type == "":
            input_ml_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_output_{feat_select}"
            input_dir = f"H:/magistro_studijos/magis/data_{dataset}/graph_output"
            output_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_optim_output_{feat_select}"

        if graph_type == "combined_metrics":
            input_ml_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_output_{feat_select}_combined_metrics"
            input_dir = f"H:/magistro_studijos/magis/data_{dataset}/graph_combined_output"
            output_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_optim_output_{feat_select}_combined_metrics"

        if graph_type == "combined_bands":
            input_ml_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_output_{feat_select}_combined_bands"
            input_dir = f"H:/magistro_studijos/magis/data_{dataset}/graph_combined_output"
            output_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_optim_output_{feat_select}_combined_bands"

    if dataset == "depression":
        if graph_type == "":
            input_ml_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_output_{feat_select}_4000_int"
            input_dir = f"H:/magistro_studijos/magis/data_{dataset}/graph_output_4000_int"
            output_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_optim_output_{feat_select}_4000_int"

        if graph_type == "combined_metrics":
            input_ml_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_output_{feat_select}_combined_metrics_4000_int"
            input_dir = f"H:/magistro_studijos/magis/data_{dataset}/graph_combined_output_4000_int"
            output_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_optim_output_{feat_select}_combined_metrics_4000_int"

        if graph_type == "combined_bands":
            input_ml_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_output_{feat_select}_combined_bands_4000_int"
            input_dir = f"H:/magistro_studijos/magis/data_{dataset}/graph_combined_output_4000_int"
            output_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_optim_output_{feat_select}_combined_bands_4000_int"


    return graph_type, input_ml_dir, input_dir, output_dir

# Step 1: Load and preprocess the data
def load_and_process_data(input_ml_dir):
    file_nms = [f for f in os.listdir(input_ml_dir) if f.endswith(".xlsx")]
    df_res = pd.DataFrame()

    pattern = r"(plv_features|pli_features|corr_features|imag_part_coh_features)_(weight|bin)_(\d+\.\d+-\d+\.\d+)\.xlsx"
    pattern_combined_metrics = r"metrics_(\d+\.\d+-\d+\.\d+).*_(bin|weight)_combined"
    pattern_combined_bands = r"(plv_features|pli_features|corr_features|imag_part_coh_features).*_(bin|weight)_combined"

    for file_nm in file_nms:
        # Read data from Excel file
        df_tmp = pd.read_excel(os.path.join(input_ml_dir, file_nm))

        # Extract information using regex
        match = re.search(pattern, file_nm)
        if match:
            feature_type, file_type, freq = match.groups()

            # Add extracted components as new columns in the DataFrame
            df_tmp['feature_type'] = feature_type
            df_tmp['type'] = file_type
            df_tmp['freq'] = freq

            # Append to the results DataFrame
            df_res = pd.concat([df_res, df_tmp], ignore_index=True)

        match_metrics = re.search(pattern_combined_metrics, file_nm)
        if match_metrics:
            freq, file_type = match_metrics.groups()

            # Add extracted components as new columns in the DataFrame
            df_tmp['type'] = file_type
            df_tmp['freq'] = freq

            # Append to the results DataFrame
            df_res = pd.concat([df_res, df_tmp], ignore_index=True)

        match_bands = re.search(pattern_combined_bands, file_nm)
        if match_bands:
            feature_type, file_type = match_bands.groups()

            # Add extracted components as new columns in the DataFrame
            df_tmp['feature_type'] = feature_type
            df_tmp['type'] = file_type

            # Append to the results DataFrame
            df_res = pd.concat([df_res, df_tmp], ignore_index=True)

    return df_res

# Step 2: Identify the best feature sets based on maximum accuracy
def get_best_feature_sets(df_res, graph_type):
    # Sort by Mean Accuracy in descending order to prioritize maximum accuracy
    df_res_sorted = df_res.sort_values(by='Mean Accuracy', ascending=False)

    # Drop duplicate groups based on 'method', 'feature_type', 'type', and 'freq', keeping the row with the highest accuracy
    if graph_type == "" or graph_type == "combined_bands":
        best_features = df_res_sorted.drop_duplicates(subset=['method', 'feature_type', 'type'])

    if graph_type == "combined_metrics":
        best_features = df_res_sorted.drop_duplicates(subset=['method', 'type', 'freq'])

    return best_features

# Step 3: Function to optimize models using GridSearchCV
def optimize_model(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

# Step 4: Prepare the ML dataset
def prep_ml_dataset(label_1_graph, label_0_graph, num_electrodes):
    # Assuming `f.flatten_adj_mat` and `f.prep_ml_dataset` are functions you have defined elsewhere
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

    return X_train_scaled, X_test_scaled, y_train, y_test

# Step 5: Train and optimize models for each feature set
def train_and_optimize_models(X_train, y_train, X_test, y_test, feature_set = None):
    results = {}

    if feature_set is not None:
        X_train_selected = X_train[:, feature_set]
        X_test_selected = X_test[:, feature_set]
    else:
        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=10)
        X_train_selected = pca.fit_transform(X_train)
        X_test_selected = pca.transform(X_test)

    for model_name, model in models.items():
        print(f"Optimizing {model_name}...")
        best_model, best_params = optimize_model(model, param_grids[model_name], X_train_selected, y_train)

        # Perform cross-validation on the training set
        cv_scores = cross_val_score(best_model, X_train_selected, y_train, cv=5, scoring='accuracy')
        mean_cv_accuracy = cv_scores.mean()
        std_cv_accuracy = cv_scores.std()

        # Evaluate on the test set
        y_pred = best_model.predict(X_test_selected)
        test_accuracy = accuracy_score(y_test, y_pred)

        if hasattr(best_model, "predict_proba"):
            # Predict probabilities on the test set
            y_scores = best_model.predict_proba(X_test_selected)[:, 1]  # Probabilities for the positive class

            # Compute ROC curve and AUC
            fpr, tpr, thresholds = roc_curve(y_test, y_scores)
            auc = roc_auc_score(y_test, y_scores)
        else:
            # If predict_proba is not available, use decision_function (for models like SVC)
            y_scores = best_model.decision_function(X_test_selected)

            # Compute ROC curve and AUC
            fpr, tpr, thresholds = roc_curve(y_test, y_scores)
            auc = roc_auc_score(y_test, y_scores)

        results[model_name] = {
            'best_model': best_model,
            'best_params': best_params,
            'cv_mean_accuracy': mean_cv_accuracy,
            'cv_std_accuracy': std_cv_accuracy,
            'test_accuracy': test_accuracy,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc
        }

    return results

# Main function to load data, identify best features, and train models
def ml_optim_pca(dataset, graph_type):
    graph_type, input_ml_dir, input_dir, output_dir = get_metadata(dataset, graph_type, "pca")

    os.makedirs(f"{output_dir}", exist_ok=True)

    # Load and preprocess data
    df_res = load_and_process_data(input_ml_dir)
    if graph_type == "" or graph_type == "combined_bands":
        # Identify the best feature sets based on maximum accuracy
        best_features = get_best_feature_sets(df_res, graph_type)
        best_features = best_features.drop(columns=['method']).drop_duplicates(subset=['feature_type', 'type'])
    if graph_type == "combined_metrics":
        # Identify the best feature sets based on maximum accuracy
        best_features = get_best_feature_sets(df_res, graph_type)
        best_features = best_features.drop(columns=['method']).drop_duplicates(subset=['type', 'freq'])

    # Iterate over each row in the best_features DataFrame
    for index, row in tqdm.tqdm(best_features.iterrows()):
        # Access column values using row['column_name']

        if graph_type == "":
            feature_type = row['feature_type']
            label = row['type']
            frequency_range = row['freq']

            filename_1 = f"{feature_type}_label_1_{label}_{frequency_range}.npy"
            filename_0 = f"{feature_type}_label_0_{label}_{frequency_range}.npy"

            label_0_graph = np.load(f"{input_dir}/{filename_0}")
            label_1_graph = np.load(f"{input_dir}/{filename_1}")
        if graph_type == "combined_metrics":
            label = row['type']
            frequency_range = row['freq']

            filename_1 = f"metrics_{frequency_range}_label_1_{label}_combined.npy"
            filename_0 = f"metrics_{frequency_range}_label_0_{label}_combined.npy"

            label_0_graph = np.load(f"{input_dir}/{filename_0}")
            label_1_graph = np.load(f"{input_dir}/{filename_1}")
        if graph_type == "combined_bands":
            feature_type = row['feature_type']
            label = row['type']

            filename_1 = f"{feature_type}_label_1_{label}_combined.npy"
            filename_0 = f"{feature_type}_label_0_{label}_combined.npy"

            label_0_graph = np.load(f"{input_dir}/{filename_0}")
            label_1_graph = np.load(f"{input_dir}/{filename_1}")

        # Assuming num_electrodes is known or can be derived from graph shape
        num_electrodes = label_1_graph.shape[0]  # Adjust based on data

        # Preprocess the data and prepare X_train, X_test, y_train, y_test
        X_train_scaled, X_test_scaled, y_train, y_test = prep_ml_dataset(label_1_graph, label_0_graph,
                                                                         num_electrodes)

        # Now optimize the models for this specific feature set
        results = train_and_optimize_models(X_train_scaled, y_train, X_test_scaled, y_test)

        # Convert the results into a DataFrame
        results_df = pd.DataFrame(results).T  # Transpose to make results align with columns

        if graph_type == "":
            # Save the results to an Excel file
            excel_filename = f"{output_dir}/{feature_type}_{label}_{frequency_range}_optimized_results.xlsx"

        if graph_type == "combined_metrics":
            # Save the results to an Excel file
            excel_filename = f"{output_dir}/metrics_{frequency_range}_{label}_combined_optimized_results.xlsx"

        if graph_type == "combined_bands":
            # Save the results to an Excel file
            excel_filename = f"{output_dir}/{feature_type}_{label}_combined_optimized_results.xlsx"


        results_df.to_excel(excel_filename, index=False)
        print(f"Results saved to {excel_filename}")



    return "All results saved."

def ml_optim_lasso(dataset, graph_type):
    graph_type, input_ml_dir, input_dir, output_dir = get_metadata(dataset, graph_type, "lasso")

    os.makedirs(f"{output_dir}", exist_ok=True)

    # Load and preprocess data
    df_res = load_and_process_data(input_ml_dir)
    if graph_type == "" or graph_type == "combined_bands":
        # Identify the best feature sets based on maximum accuracy
        best_features = get_best_feature_sets(df_res, graph_type)
        best_features = best_features.drop(columns=['method']).drop_duplicates(subset=['feature_type', 'type'])
    if graph_type == "combined_metrics":
        # Identify the best feature sets based on maximum accuracy
        best_features = get_best_feature_sets(df_res, graph_type)
        best_features = best_features.drop(columns=['method']).drop_duplicates(subset=['type', 'freq'])

    # Iterate over each row in the best_features DataFrame
    for index, row in tqdm.tqdm(best_features.iterrows()):
        # Access column values using row['column_name']

        if graph_type == "":
            feature_type = row['feature_type']
            label = row['type']
            frequency_range = row['freq']
            var_features = ast.literal_eval(row['selected features'])

            filename_1 = f"{feature_type}_label_1_{label}_{frequency_range}.npy"
            filename_0 = f"{feature_type}_label_0_{label}_{frequency_range}.npy"

            label_0_graph = np.load(f"{input_dir}/{filename_0}")
            label_1_graph = np.load(f"{input_dir}/{filename_1}")
        if graph_type == "combined_metrics":
            label = row['type']
            frequency_range = row['freq']
            var_features = ast.literal_eval(row['selected features'])

            filename_1 = f"metrics_{frequency_range}_label_1_{label}_combined.npy"
            filename_0 = f"metrics_{frequency_range}_label_0_{label}_combined.npy"

            label_0_graph = np.load(f"{input_dir}/{filename_0}")
            label_1_graph = np.load(f"{input_dir}/{filename_1}")
        if graph_type == "combined_bands":
            feature_type = row['feature_type']
            label = row['type']
            var_features = ast.literal_eval(row['selected features'])

            filename_1 = f"{feature_type}_label_1_{label}_combined.npy"
            filename_0 = f"{feature_type}_label_0_{label}_combined.npy"

            label_0_graph = np.load(f"{input_dir}/{filename_0}")
            label_1_graph = np.load(f"{input_dir}/{filename_1}")

        # Assuming num_electrodes is known or can be derived from graph shape
        num_electrodes = label_1_graph.shape[0]  # Adjust based on data

        # Preprocess the data and prepare X_train, X_test, y_train, y_test
        X_train_scaled, X_test_scaled, y_train, y_test = prep_ml_dataset(label_1_graph, label_0_graph,
                                                                         num_electrodes)

        # Now optimize the models for this specific feature set
        print(f"Training and optimizing for feature set: {var_features}")
        results = train_and_optimize_models(X_train_scaled, y_train, X_test_scaled, y_test, var_features)

        # Convert the results into a DataFrame
        results_df = pd.DataFrame(results).T  # Transpose to make results align with columns

        if graph_type == "":
            # Save the results to an Excel file
            excel_filename = f"{output_dir}/{feature_type}_{label}_{frequency_range}_optimized_results.xlsx"

        if graph_type == "combined_metrics":
            # Save the results to an Excel file
            excel_filename = f"{output_dir}/metrics_{frequency_range}_{label}_combined_optimized_results.xlsx"

        if graph_type == "combined_bands":
            # Save the results to an Excel file
            excel_filename = f"{output_dir}/{feature_type}_{label}_combined_optimized_results.xlsx"


        results_df.to_excel(excel_filename, index=False)
        print(f"Results saved to {excel_filename}")



    return "All results saved."

# Run the main function
if __name__ == '__main__':
    ml_optim_pca(dataset, graph_type)
    ml_optim_lasso(dataset, graph_type)


import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import functions as f
import tqdm
import ast

# Define the directories
input_ml_dir = "H:/magistro_studijos/magis/data_eyes/ml_output"
input_dir = "H:/magistro_studijos/magis/data_eyes/graph_output"
output_dir = "H:/magistro_studijos/magis/data_eyes/ml_optim_output"
pattern = r"(plv_features|pli_features|corr_features|imag_part_coh_features)_(weight|bin)_(\d+\.\d+-\d+\.\d+)\.xlsx"

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

# Step 1: Load and preprocess the data
def load_and_process_data(input_ml_dir, pattern):
    file_nms = [f for f in os.listdir(input_ml_dir) if f.endswith(".xlsx")]
    df_res = pd.DataFrame()

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

    return df_res

# Step 2: Identify the best feature sets based on maximum accuracy
def get_best_feature_sets(df_res):
    # Sort by Mean Accuracy in descending order to prioritize maximum accuracy
    df_res_sorted = df_res.sort_values(by='Mean Accuracy', ascending=False)

    # Drop duplicate groups based on 'method', 'feature_type', 'type', and 'freq', keeping the row with the highest accuracy
    best_features = df_res_sorted.drop_duplicates(subset=['method', 'feature_type', 'type', 'freq'])

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
def train_and_optimize_models(X_train, y_train, X_test, y_test, feature_set):
    results = {}

    X_train_selected = X_train[:, feature_set]
    X_test_selected = X_test[:, feature_set]

    for model_name, model in models.items():
        print(f"Optimizing {model_name}...")
        best_model, best_params = optimize_model(model, param_grids[model_name], X_train_selected, y_train)
        y_pred = best_model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)

        results[model_name] = {
            'best_model': best_model,
            'best_params': best_params,
            'accuracy': accuracy
        }

    return results

# Main function to load data, identify best features, and train models
def main(pattern):
    # Load and preprocess data
    df_res = load_and_process_data(input_ml_dir, pattern)

    # Identify the best feature sets based on maximum accuracy
    best_features = get_best_feature_sets(df_res)
    best_features = best_features.drop(columns=['method']).drop_duplicates()

    # Iterate over each row in the best_features DataFrame
    for index, row in tqdm.tqdm(best_features.iterrows()):
        # Access column values using row['column_name']

        feature_type = row['feature_type']
        label = row['type']
        frequency_range = row['freq']
        var_features = ast.literal_eval(row['selected features'])

        filename_1 = f"{feature_type}_label_1_{label}_{frequency_range}.npy"
        filename_0 = f"{feature_type}_label_0_{label}_{frequency_range}.npy"

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

        # Save the results to an Excel file
        excel_filename = f"{output_dir}/{feature_type}_{label}_{frequency_range}_optimized_results.xlsx"
        results_df.to_excel(excel_filename, index=False)
        print(f"Results saved to {excel_filename}")



    return "All results saved."


# Run the main function
if __name__ == '__main__':
    print(all_results = main(pattern))


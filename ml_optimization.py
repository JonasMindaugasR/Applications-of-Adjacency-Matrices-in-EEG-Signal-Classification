import functions as f
import pandas as pd
import tqdm
import os
import re

dataset = "eyes" # "eyes" "depression"

input_dir = f"H:/magistro_studijos/magis/data_{dataset}/output"
output_dir = f"H:/magistro_studijos/magis/data_{dataset}/ml_output"

num_electrodes = 64

fs = 120
int_start = 4500
int_end = 5000

if __name__ == '__main__':
    filenames = os.listdir(input_dir)
    for filename in tqdm.tqdm(filenames):
        # Use regex to find the frequency range pattern
        match = re.search(r'(\d+\.?\d*)-(\d+\.?\d*)', filename)
        if match:
            results_df = pd.DataFrame(columns=[
                'interval start',
                'interval end',
                'normalized trial',
                'l_freq',
                'h_freq',
                'binarized',
                'adj_feature',
                'method',
                'Mean Accuracy',
                'Mean F1 Score',
                '95% CI for Accuracy Low',
                '95% CI for Accuracy High',
                '95% CI for F1 Score Low',
                '95% CI for F1 Score High'
            ])
            for normalize in [True, False]:
                opened_raw, closed_raw = f.data_preparation(path=f"{input_dir}/{filename}", int_start = int_start, int_end = int_end, normalize=normalize)

                opened, closed = f.filter_dataset(opened_raw, closed_raw, float(match.group(1)), float(match.group(2)), fs)

                for binarize in [True, False]:
                    for upp_triangle in [True, False]:
                        for feature in [f.plv_features, f.corr_features, f.pli_features]:
                            # calculate correlation matrix for one observation and append it to the list
                            opened_corr, closed_corr = f.calc_adj_features(opened, closed, feature, binarize=binarize, threshold=0, num_electrodes=num_electrodes)

                            opened_corr_flat, closed_corr_flat = f.flatten_adj_mat(opened_corr, closed_corr, upp_triangle=upp_triangle,  num_electrodes=num_electrodes)

                            X_train_scaled, X_test_scaled, y_train, y_test = f.prep_ml_dataset(opened_corr_flat, closed_corr_flat, metric=False)

                            selected_features = f.lasso_optimization(X_train_scaled, y_train, alpha=0.1)

                            for model in [f.svm_lasso_bootsrap, f.rf_lasso_bootsrap, f.xgb_lasso_bootsrap]:

                                method, mean_accuracy, mean_f1_score, ci_accuracy, ci_f1_score = model(X_train_scaled, X_test_scaled, y_train, y_test, selected_features)

                                # Create a new row as a DataFrame
                                new_row = pd.DataFrame({
                                    'interval start': [int_start],
                                    'interval end': [int_end],
                                    'normalized trial': [normalize],
                                    'l_freq': [float(match.group(1))],
                                    'h_freq': [float(match.group(2))],
                                    'binarized': [binarize],
                                    'adj_feature': [feature.__name__],
                                    'method': [method],
                                    'Mean Accuracy': [mean_accuracy],
                                    'Mean F1 Score': [mean_f1_score],
                                    '95% CI for Accuracy Low': [ci_accuracy[0]],
                                    '95% CI for Accuracy High': [ci_accuracy[1]],
                                    '95% CI for F1 Score Low': [ci_f1_score[0]],
                                    '95% CI for F1 Score High': [ci_f1_score[1]]
                                })


                                # Append the new row to the DataFrame
                                results_df = pd.concat([results_df, new_row], ignore_index=False)

    # Save DataFrame as Excel
    results_df.to_excel(f"{output_dir}/df_result.xlsx", index=False)
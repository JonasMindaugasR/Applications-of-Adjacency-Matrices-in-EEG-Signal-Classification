rm(list = ls())

set.seed(123)

library(data.table)
library(ggplot2)
library(ggpubr)
library(patchwork)
library(svglite)
library(plotly)
library(caret)
library(randomForest)
library(DescTools)
library(car)
library(iml)
library(dplyr)
library(randomForestExplainer)

#----Functions----

# join tables from all folder function
GetData <- function(dir, optim) {

  file.nms.no.select <- list.files(dir)[grep("no_select", list.files(dir))]
  file.nms.select <- list.files(dir)[!list.files(dir) %in% file.nms.no.select]

  # Regular expression to extract plv_features, label, and interval
  pattern.select <- "(plv_features|pli_features|corr_features|imag_part_coh_features)_(weight|bin)_(\\d+\\.\\d+-\\d+\\.\\d+)\\_optimized_results.xlsx"

  pattern.no.select <- "(plv_features|pli_features|corr_features|imag_part_coh_features)_(weight|bin)_(\\d+\\.\\d+-\\d+\\.\\d+)\\_no_select_optimized_results.xlsx"

  dt.res <- data.table()

  for (file.nm in file.nms.select) {
    dt.tmp <- as.data.table(readxl::read_xlsx(paste0(dir, file.nm)))

    # Use regexec to find matches
    matches <- regexec(pattern.select, file.nm)

    # Extract the matched components using regmatches
    matches <- regmatches(file.nm, matches)

    # add params
    dt.tmp[, feature_type := matches[[1]][2]]
    dt.tmp[, type := matches[[1]][3]]
    dt.tmp[, freq := matches[[1]][4]]

    dt.res <- rbind(dt.res, dt.tmp)
  }

  dt.res.no.select <- data.table()

  for (file.nm in file.nms.no.select) {
    dt.tmp <- as.data.table(readxl::read_xlsx(paste0(dir, file.nm)))

    # Use regexec to find matches
    matches <- regexec(pattern.no.select, file.nm)

    # Extract the matched components using regmatches
    matches <- regmatches(file.nm, matches)

    # add params
    dt.tmp[, feature_type := matches[[1]][2]]
    dt.tmp[, type := matches[[1]][3]]
    dt.tmp[, freq := matches[[1]][4]]

    dt.res.no.select <- rbind(dt.res.no.select, dt.tmp)
  }

  dt.res.no.select[, optim := "Var"]
  dt.res[, optim := optim]

  dt.res <- rbind(dt.res.no.select, dt.res)

  dt.res[, model := gsub("\\s*\\([^)]*\\)", "", best_model)]

  return(dt.res)
}

GetDataBands <- function(dir, optim) {

  file.nms.no.select <- list.files(dir)[grep("no_select", list.files(dir))]
  file.nms.select <- list.files(dir)[!list.files(dir) %in% file.nms.no.select]

  # Regular expression to extract plv_features, label, and interval
  pattern.select <- "(plv_features|pli_features|corr_features|imag_part_coh_features)_(weight|bin)_combined_optimized_results.xlsx"

  pattern.no.select <- "(plv_features|pli_features|corr_features|imag_part_coh_features)_(weight|bin)_no_select_combined_optimized_results.xlsx"

  dt.res <- data.table()

  for (file.nm in file.nms.select) {
    dt.tmp <- as.data.table(readxl::read_xlsx(paste0(dir, file.nm)))

    # Use regexec to find matches
    matches <- regexec(pattern.select, file.nm)

    # Extract the matched components using regmatches
    matches <- regmatches(file.nm, matches)

    # add params
    dt.tmp[, feature_type := matches[[1]][2]]
    dt.tmp[, type := matches[[1]][3]]
    dt.tmp[, freq := matches[[1]][4]]

    dt.res <- rbind(dt.res, dt.tmp)
  }

  dt.res.no.select <- data.table()

  for (file.nm in file.nms.no.select) {
    dt.tmp <- as.data.table(readxl::read_xlsx(paste0(dir, file.nm)))

    # Use regexec to find matches
    matches <- regexec(pattern.no.select, file.nm)

    # Extract the matched components using regmatches
    matches <- regmatches(file.nm, matches)

    # add params
    dt.tmp[, feature_type := matches[[1]][2]]
    dt.tmp[, type := matches[[1]][3]]
    dt.tmp[, freq := matches[[1]][4]]

    dt.res.no.select <- rbind(dt.res.no.select, dt.tmp)
  }

  dt.res.no.select[, optim := "Var"]
  dt.res[, optim := optim]

  dt.res <- rbind(dt.res.no.select, dt.res)

  dt.res[, model := gsub("\\s*\\([^)]*\\)", "", best_model)]

  return(dt.res)
}

GetDataMetrics <- function(dir, optim) {

  file.nms.no.select <- list.files(dir)[grep("no_select", list.files(dir))]
  file.nms.select <- list.files(dir)[!list.files(dir) %in% file.nms.no.select]

  # Regular expression to extract plv_features, label, and interval
  pattern.select <- "metrics_(\\d+\\.\\d+-\\d+\\.\\d+)\\_(weight|bin)_combined_optimized_results.xlsx"

  pattern.no.select <- "metrics_(\\d+\\.\\d+-\\d+\\.\\d+)\\_(weight|bin)_no_select_combined_optimized_results.xlsx"

  dt.res <- data.table()

  for (file.nm in file.nms.select) {
    dt.tmp <- as.data.table(readxl::read_xlsx(paste0(dir, file.nm)))

    # Use regexec to find matches
    matches <- regexec(pattern.select, file.nm)

    # Extract the matched components using regmatches
    matches <- regmatches(file.nm, matches)

    # add params
    dt.tmp[, feature_type := matches[[1]][2]]
    dt.tmp[, type := matches[[1]][3]]
    dt.tmp[, freq := matches[[1]][4]]

    dt.res <- rbind(dt.res, dt.tmp)
  }

  dt.res.no.select <- data.table()

  for (file.nm in file.nms.no.select) {
    dt.tmp <- as.data.table(readxl::read_xlsx(paste0(dir, file.nm)))

    # Use regexec to find matches
    matches <- regexec(pattern.no.select, file.nm)

    # Extract the matched components using regmatches
    matches <- regmatches(file.nm, matches)

    # add params
    dt.tmp[, feature_type := matches[[1]][2]]
    dt.tmp[, type := matches[[1]][3]]
    dt.tmp[, freq := matches[[1]][4]]

    dt.res.no.select <- rbind(dt.res.no.select, dt.tmp)
  }

  dt.res.no.select[, optim := "Var"]
  dt.res[, optim := optim]

  dt.res <- rbind(dt.res.no.select, dt.res)

  dt.res[, model := gsub("\\s*\\([^)]*\\)", "", best_model)]

  return(dt.res)
}

GetDataAll <- function(dir, optim) {

  file.nms.no.select <- list.files(dir)[grep("no_select", list.files(dir))]
  file.nms.select <- list.files(dir)[!list.files(dir) %in% file.nms.no.select]

  # Regular expression to extract plv_features, label, and interval
  pattern.select <- "metrics_freq_(weight|bin)_combined_optimized_results.xlsx"

  pattern.no.select <- "metrics_freq_(weight|bin)_no_select_combined_optimized_results.xlsx"

  dt.res <- data.table()

  for (file.nm in file.nms.select) {
    dt.tmp <- as.data.table(readxl::read_xlsx(paste0(dir, file.nm)))

    # Use regexec to find matches
    matches <- regexec(pattern.select, file.nm)

    # Extract the matched components using regmatches
    matches <- regmatches(file.nm, matches)

    # add params
    dt.tmp[, type := matches[[1]][2]]

    dt.res <- rbind(dt.res, dt.tmp)
  }

  dt.res.no.select <- data.table()

  for (file.nm in file.nms.no.select) {
    dt.tmp <- as.data.table(readxl::read_xlsx(paste0(dir, file.nm)))

    # Use regexec to find matches
    matches <- regexec(pattern.no.select, file.nm)

    # Extract the matched components using regmatches
    matches <- regmatches(file.nm, matches)

    # add params
    dt.tmp[, type := matches[[1]][2]]

    dt.res.no.select <- rbind(dt.res.no.select, dt.tmp)
  }

  if (nrow(dt.res.no.select) > 0) {dt.res.no.select[, optim := "Var"]}
  dt.res[, optim := optim]

  dt.res <- rbind(dt.res.no.select, dt.res)

  dt.res[, model := gsub("\\s*\\([^)]*\\)", "", best_model)]

  return(dt.res)
}

GetDatasetResults <- function(dataset){
    # non aggregated
    dir_depr <- paste0("H:/magistro_studijos/magis/final_results_3/", dataset,"/")
    dir_lasso <- paste0(dir_depr, "ml_optim_output_lasso/")
    dt.depr <- GetData(dir_lasso, "lasso")
    dir_pca <- paste0(dir_depr, "ml_optim_output_pca/")
    dt.depr.pca <- GetData(dir_pca, "pca")
    dt.depr.pca[optim == "Var", optim := NA]
    dt.depr.pca <- dt.depr.pca[!is.na(optim)]
    dt.depr <- rbind(dt.depr, dt.depr.pca)
    dt_res_non_agg <- dt.depr[, .(type, optim, freq, feature_type, model, test_accuracy, auc)]

    rm(dt.depr)

    dt_res_non_agg[, agg := "Non aggregated"]

    # agg on metrics
    dir_depr <- paste0("H:/magistro_studijos/magis/final_results_3/", dataset,"/")
    dir_lasso <- paste0(dir_depr, "ml_optim_output_lasso_combined_metrics/")
    dt.depr <- GetDataMetrics(dir_lasso, "lasso")
    dir_pca <- paste0(dir_depr, "ml_optim_output_pca_combined_metrics/")
    dt.depr.pca <- GetDataMetrics(dir_pca, "pca")
    dt.depr.pca[optim == "Var", optim := NA]
    dt.depr.pca <- dt.depr.pca[!is.na(optim)]
    dt.depr <- rbind(dt.depr, dt.depr.pca)
    dt_res_agg_metrics <- dt.depr[, .(type, optim, feature_type, model, test_accuracy, auc)]
    setnames(dt_res_agg_metrics, "feature_type", "freq")

    rm(dt.depr)

    dt_res_agg_metrics[, agg := "Aggregated on metrics"]

    # agg on bands
    dir_depr <- paste0("H:/magistro_studijos/magis/final_results_3/", dataset,"/")
    dir_lasso <- paste0(dir_depr, "ml_optim_output_lasso_combined_bands/")
    dt.depr <- GetDataBands(dir_lasso, "lasso")
    dir_pca <- paste0(dir_depr, "ml_optim_output_pca_combined_bands/")
    dt.depr.pca <- GetDataBands(dir_pca, "pca")
    dt.depr.pca[optim == "Var", optim := NA]
    dt.depr.pca <- dt.depr.pca[!is.na(optim)]
    dt.depr <- rbind(dt.depr, dt.depr.pca)
    dt_res_agg_bands <- dt.depr[, .(type, optim, feature_type, model, test_accuracy, auc)]

    rm(dt.depr)

    dt_res_agg_bands[, agg := "Aggregated on bands"]

    # fully agg
    dir_depr <- paste0("H:/magistro_studijos/magis/final_results_3/", dataset,"/")
    dir_lasso <- paste0(dir_depr, "ml_optim_output_lasso_combined_all/")
    dt.depr <- GetDataAll(dir_lasso, "lasso")
    dir_pca <- paste0(dir_depr, "ml_optim_output_pca_combined_all/")
    dt.depr.pca <- GetDataAll(dir_pca, "pca")
    dt.depr.pca[optim == "Var", optim := NA]
    dt.depr.pca <- dt.depr.pca[!is.na(optim)]
    dt.depr <- rbind(dt.depr, dt.depr.pca)
    dt_res_agg_all <- dt.depr[, .(type, optim, model, test_accuracy, auc)]

    rm(dt.depr)

    dt_res_agg_all[, agg := "Aggregated fully"]

    combined_results <- rbindlist(
      list(dt_res_non_agg, dt_res_agg_bands, dt_res_agg_metrics, dt_res_agg_all),
      fill = TRUE
    )

    return(combined_results)
}

PlotData <- function(dir) {

  dir_lasso <- paste0(dir, "ml_optim_output_lasso/")

  dt.depr <- GetData(dir_lasso, "lasso")

  # dt.depr <- data.table()

  dir_pca <- paste0(dir, "ml_optim_output_pca/")

  dt.depr.tmp <- GetData(dir_pca, "pca")

  dt.depr.tmp[optim == "Var", optim := NA]
  dt.depr.tmp <- dt.depr.tmp[!is.na(optim)]

  dt.depr <- rbind(dt.depr, dt.depr.tmp)

  dt.depr[model == "SVC", model := "SVM"]
  dt.depr[model == "RandomForestClassifier", model := "Random Forest"]
  dt.depr[model == "XGBClassifier", model := "XGBoost"]

  dt.depr[feature_type == "corr_features", feature_type := "Corr"]
  dt.depr[feature_type == "plv_features", feature_type := "PLV"]
  dt.depr[feature_type == "pli_features", feature_type := "PLI"]
  dt.depr[feature_type == "imag_part_coh_features", feature_type := "iCoh"]

  dt.depr[optim == "lasso", optim := "LASSO"]
  dt.depr[optim == "pca", optim := "PCA"]
  dt.depr[optim == "Var", optim := "Variance"]

  dt.depr[, test_accuracy := test_accuracy*100]
  dt.depr[, auc := auc*100]

  dt.depr$optim <- factor(dt.depr$optim, levels = c("Variance", "LASSO", "PCA"))

  dt.depr$freq <- factor(dt.depr$freq, levels = c("0.5-40.0", "0.5-4.0", "4.0-8.0", "8.0-12.0", "12.0-30.0", "30.0-40.0"))

  setnames(dt.depr, c("model", "feature_type"), c("Model", "Adjacency metric"))

  bin_plot_all <- ggplot(dt.depr[type == "bin"][freq == "0.5-40.0"],
                         aes(x = test_accuracy, y = auc, color = `Adjacency metric`, shape = Model)) +
      geom_point(size = 4, alpha = 0.8) +  # Larger points with slight transparency for better clarity
      theme_minimal(base_size = 14) +     # Minimal theme with larger base font size
      theme(
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  # Center and bold the title
        axis.title = element_text(size = 14, face = "bold"),              # Bold axis titles
        axis.text = element_text(size = 12),                              # Larger axis text
        legend.title = element_text(size = 13, face = "bold"),            # Bold legend titles
        legend.text = element_text(size = 11),                            # Larger legend text
        strip.text = element_text(size = 13, face = "bold")               # Bold facet labels
      ) +
      scale_color_brewer(palette = "Set1") +  # Use a visually appealing color palette
      scale_shape_manual(values = c(16, 17, 18, 19)) +  # Assign consistent shapes
      labs(
        x = "Model Accuracy (%)",
        y = "Area Under Curve (AUC)",
        color = "Adjacency Metric",
        shape = "Model Type"
      ) +
      facet_grid(freq ~ optim)

  bin_plot_freq <- ggplot(dt.depr[type == "bin"][freq != "0.5-40.0"],
                          aes(x = test_accuracy, y = auc, color = `Adjacency metric`, shape = Model)) +
      geom_point(size = 4, alpha = 0.8) +  # Larger points with slight transparency for better clarity
      theme_minimal(base_size = 14) +     # Minimal theme with larger base font size
      theme(
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  # Center and bold the title
        axis.title = element_text(size = 14, face = "bold"),              # Bold axis titles
        axis.text = element_text(size = 12),                              # Larger axis text
        legend.title = element_text(size = 13, face = "bold"),            # Bold legend titles
        legend.text = element_text(size = 11),                            # Larger legend text
        strip.text = element_text(size = 13, face = "bold")               # Bold facet labels
      ) +
      scale_color_brewer(palette = "Set1") +  # Use a visually appealing color palette
      scale_shape_manual(values = c(16, 17, 18, 19)) +  # Assign consistent shapes
      labs(
        x = "Model Accuracy (%)",
        y = "Area Under Curve (AUC)",
        color = "Adjacency Metric",
        shape = "Model Type"
      ) +
      facet_grid(freq ~ optim)

  weight_plot_all <- ggplot(dt.depr[type == "weight"][freq == "0.5-40.0"],
                            aes(x = test_accuracy, y = auc, color = `Adjacency metric`, shape = Model)) +
      geom_point(size = 4, alpha = 0.8) +  # Larger points with slight transparency for better clarity
      theme_minimal(base_size = 14) +     # Minimal theme with larger base font size
      theme(
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  # Center and bold the title
        axis.title = element_text(size = 14, face = "bold"),              # Bold axis titles
        axis.text = element_text(size = 12),                              # Larger axis text
        legend.title = element_text(size = 13, face = "bold"),            # Bold legend titles
        legend.text = element_text(size = 11),                            # Larger legend text
        strip.text = element_text(size = 13, face = "bold")               # Bold facet labels
      ) +
      scale_color_brewer(palette = "Set1") +  # Use a visually appealing color palette
      scale_shape_manual(values = c(16, 17, 18, 19)) +  # Assign consistent shapes
      labs(
        x = "Model Accuracy (%)",
        y = "Area Under Curve (AUC)",
        color = "Adjacency Metric",
        shape = "Model Type"
      ) +
      facet_grid(freq ~ optim)

  weight_plot_freq <- ggplot(dt.depr[type == "weight"][freq != "0.5-40.0"],
                             aes(x = test_accuracy, y = auc, color = `Adjacency metric`, shape = Model)) +
      geom_point(size = 4, alpha = 0.8) +  # Larger points with slight transparency for better clarity
      theme_minimal(base_size = 14) +     # Minimal theme with larger base font size
      theme(
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  # Center and bold the title
        axis.title = element_text(size = 14, face = "bold"),              # Bold axis titles
        axis.text = element_text(size = 12),                              # Larger axis text
        legend.title = element_text(size = 13, face = "bold"),            # Bold legend titles
        legend.text = element_text(size = 11),                            # Larger legend text
        strip.text = element_text(size = 13, face = "bold")               # Bold facet labels
      ) +
      scale_color_brewer(palette = "Set1") +  # Use a visually appealing color palette
      scale_shape_manual(values = c(16, 17, 18, 19)) +  # Assign consistent shapes
      labs(
        x = "Model Accuracy (%)",
        y = "Area Under Curve (AUC)",
        color = "Adjacency Metric",
        shape = "Model Type"
      ) +
      facet_grid(freq ~ optim)

  res <- list("binarized_all" = bin_plot_all,
              "weighted_all" = weight_plot_all,
              "binarized_freq" = bin_plot_freq,
              "weighted_freq" = weight_plot_freq)

  return(res)
}

PlotDataCV <- function(dir) {

  # dir_lasso <- paste0(dir, "ml_optim_output_lasso/")
  #
  # dt.depr <- GetData(dir_lasso, "lasso")

  dt.depr <- data.table()

  dir_pca <- paste0(dir, "ml_optim_output_pca_final/")

  dt.depr.tmp <- GetData(dir_pca, "pca")

  # dt.depr.tmp[optim == "Var", optim := NA]
  # dt.depr.tmp <- dt.depr.tmp[!is.na(optim)]

  dt.depr <- rbind(dt.depr, dt.depr.tmp)

  dt.depr[model == "SVC", model := "SVM"]
  dt.depr[model == "RandomForestClassifier", model := "Random Forest"]
  dt.depr[model == "XGBClassifier", model := "XGBoost"]

  dt.depr[feature_type == "corr_features", feature_type := "Corr"]
  dt.depr[feature_type == "plv_features", feature_type := "PLV"]
  dt.depr[feature_type == "pli_features", feature_type := "PLI"]
  dt.depr[feature_type == "imag_part_coh_features", feature_type := "iCoh"]

  dt.depr[optim == "lasso", optim := "LASSO"]
  dt.depr[optim == "pca", optim := "PCA"]
  dt.depr[optim == "Var", optim := "Variance"]

  dt.depr$optim <- factor(dt.depr$optim, levels = c("Variance", "LASSO", "PCA"))

  dt.depr$freq <- factor(dt.depr$freq, levels = c("0.5-40.0", "0.5-4.0", "4.0-8.0", "8.0-12.0", "12.0-30.0", "30.0-40.0"))

  setnames(dt.depr, c("model", "feature_type"), c("Model", "Adjacency metric"))

  bin_plot_all <- ggplot(dt.depr[type == "bin"][freq == "0.5-40.0"],
                         aes(x = cv_mean_accuracy, y = cv_mean_auc, color = `Adjacency metric`, shape = Model)) +
      geom_point(size = 3) +         # Add points
      theme_minimal() +             # Use a minimal theme
      labs(
        title = "Model performance by feature extraction and selection type - binarized.",
        x = "Model Accuracy",
        y = "AUC"
      )+
      # geom_hline(yintercept = 0.7, linetype = "dashed", color = "red", size = 1) +  # horizontal line at y = 0.7
      # geom_vline(xintercept = 0.7, linetype = "dashed", color = "red", size = 1) +  # vertical line at x = 0.7
      facet_grid(freq ~ optim)

  bin_plot_freq <- ggplot(dt.depr[type == "bin"][freq != "0.5-40.0"],
                          aes(x = cv_mean_accuracy, y = cv_mean_auc, color = `Adjacency metric`, shape = Model)) +
      geom_point(size = 3) +         # Add points
      theme_minimal() +             # Use a minimal theme
      labs(
        title = "Model performance by feature extraction and selection type - binarized.",
        x = "Model Accuracy",
        y = "AUC"
      )+
      # geom_hline(yintercept = 0.7, linetype = "dashed", color = "red", size = 1) +  # horizontal line at y = 0.7
      # geom_vline(xintercept = 0.7, linetype = "dashed", color = "red", size = 1) +  # vertical line at x = 0.7
      facet_grid(freq ~ optim)

  weight_plot_all <- ggplot(dt.depr[type == "weight"][freq == "0.5-40.0"],
                            aes(x = cv_mean_accuracy, y = cv_mean_auc, color = `Adjacency metric`, shape = Model)) +
      geom_point(size = 3) +         # Add points
      theme_minimal() +             # Use a minimal theme
      labs(
        title = "Model performance by feature extraction and selection type - weighted.",
        x = "Model Accuracy",
        y = "AUC"
      )+
      # geom_hline(yintercept = 0.7, linetype = "dashed", color = "red", size = 1) +  # horizontal line at y = 0.7
      # geom_vline(xintercept = 0.7, linetype = "dashed", color = "red", size = 1) +  # vertical line at x = 0.7
      facet_grid(freq ~ optim)

  weight_plot_freq <- ggplot(dt.depr[type == "weight"][freq != "0.5-40.0"],
                             aes(x = cv_mean_accuracy, y = cv_mean_auc, color = `Adjacency metric`, shape = Model)) +
      geom_point(size = 3) +         # Add points
      theme_minimal() +             # Use a minimal theme
      labs(
        title = "Model performance by feature extraction and selection type - weighted.",
        x = "Model Accuracy",
        y = "AUC"
      )+
      # geom_hline(yintercept = 0.7, linetype = "dashed", color = "red", size = 1) +  # horizontal line at y = 0.7
      # geom_vline(xintercept = 0.7, linetype = "dashed", color = "red", size = 1) +  # vertical line at x = 0.7
      facet_grid(freq ~ optim)

  res <- list("binarized_all" = bin_plot_all,
              "weighted_all" = weight_plot_all,
              "binarized_freq" = bin_plot_freq,
              "weighted_freq" = weight_plot_freq)

  return(res)
}

PlotDataBands <- function(dir) {

  dir_lasso <- paste0(dir, "ml_optim_output_lasso_combined_bands/")

  dt.depr.bands <- GetDataBands(dir_lasso, "lasso")

  dir_pca <- paste0(dir, "ml_optim_output_pca_combined_bands/")

  dt.depr.bands.tmp <- GetDataBands(dir_pca, "pca")

  dt.depr.bands.tmp[optim == "Var", optim := NA]
  dt.depr.bands.tmp <- dt.depr.bands.tmp[!is.na(optim)]

  dt.depr.bands <- rbind(dt.depr.bands, dt.depr.bands.tmp)

  dt.depr.bands[model == "SVC", model := "SVM"]
  dt.depr.bands[model == "RandomForestClassifier", model := "Random Forest"]
  dt.depr.bands[model == "XGBClassifier", model := "XGBoost"]

  dt.depr.bands[feature_type == "corr_features", feature_type := "Corr"]
  dt.depr.bands[feature_type == "plv_features", feature_type := "PLV"]
  dt.depr.bands[feature_type == "pli_features", feature_type := "PLI"]
  dt.depr.bands[feature_type == "imag_part_coh_features", feature_type := "iCoh"]

  dt.depr.bands[optim == "lasso", optim := "LASSO"]
  dt.depr.bands[optim == "pca", optim := "PCA"]
  dt.depr.bands[optim == "Var", optim := "Variance"]

  dt.depr.bands[, test_accuracy := test_accuracy*100]
  dt.depr.bands[, auc := auc*100]

  dt.depr.bands$optim <- factor(dt.depr.bands$optim, levels = c("Variance", "LASSO", "PCA"))

  setnames(dt.depr.bands, c("model", "feature_type"), c("Model", "Adjacency metric"))

bin_plot <- ggplot(dt.depr.bands[type == "bin"],
                   aes(x = test_accuracy, y = auc, color = `Adjacency metric`, shape = Model)) +
    geom_point(size = 4, alpha = 0.8) +  # Larger points with slight transparency for better clarity
      theme_minimal(base_size = 14) +     # Minimal theme with larger base font size
      theme(
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  # Center and bold the title
        axis.title = element_text(size = 14, face = "bold"),              # Bold axis titles
        axis.text = element_text(size = 12),                              # Larger axis text
        legend.title = element_text(size = 13, face = "bold"),            # Bold legend titles
        legend.text = element_text(size = 11),                            # Larger legend text
        strip.text = element_text(size = 13, face = "bold")               # Bold facet labels
      ) +
      scale_color_brewer(palette = "Set1") +  # Use a visually appealing color palette
      scale_shape_manual(values = c(16, 17, 18, 19)) +  # Assign consistent shapes
      labs(
        x = "Model Accuracy (%)",
        y = "Area Under Curve (AUC)",
        color = "Adjacency Metric",
        shape = "Model Type"
      ) +
      facet_grid(. ~ optim)

weight_plot <- ggplot(dt.depr.bands[type == "weight"],
                      aes(x = test_accuracy, y = auc, color = `Adjacency metric`, shape = Model)) +
    geom_point(size = 4, alpha = 0.8) +  # Larger points with slight transparency for better clarity
      theme_minimal(base_size = 14) +     # Minimal theme with larger base font size
      theme(
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  # Center and bold the title
        axis.title = element_text(size = 14, face = "bold"),              # Bold axis titles
        axis.text = element_text(size = 12),                              # Larger axis text
        legend.title = element_text(size = 13, face = "bold"),            # Bold legend titles
        legend.text = element_text(size = 11),                            # Larger legend text
        strip.text = element_text(size = 13, face = "bold")               # Bold facet labels
      ) +
      scale_color_brewer(palette = "Set1") +  # Use a visually appealing color palette
      scale_shape_manual(values = c(16, 17, 18, 19)) +  # Assign consistent shapes
      labs(
        x = "Model Accuracy (%)",
        y = "Area Under Curve (AUC)",
        color = "Adjacency Metric",
        shape = "Model Type"
      ) +
      facet_grid(. ~ optim)

  res <- list("binarized" = bin_plot,
              "weighted" = weight_plot)

  return(res)
}

PlotDataMetrics <- function(dir) {

  dir_lasso <- paste0(dir, "ml_optim_output_lasso_combined_metrics/")

  dt.depr <- GetDataMetrics(dir_lasso, "lasso")

  dir_pca <- paste0(dir, "ml_optim_output_pca_combined_metrics/")

  dt.depr.tmp <- GetDataMetrics(dir_pca, "pca")

  dt.depr.tmp[optim == "Var", optim := NA]
  dt.depr.tmp <- dt.depr.tmp[!is.na(optim)]

  dt.depr <- rbind(dt.depr, dt.depr.tmp)

  dt.depr[model == "SVC", model := "SVM"]
  dt.depr[model == "RandomForestClassifier", model := "Random Forest"]
  dt.depr[model == "XGBClassifier", model := "XGBoost"]

  dt.depr[optim == "lasso", optim := "LASSO"]
  dt.depr[optim == "pca", optim := "PCA"]
  dt.depr[optim == "Var", optim := "Variance"]

  dt.depr[, test_accuracy := test_accuracy*100]
  dt.depr[, auc := auc*100]

  dt.depr$optim <- factor(dt.depr$optim, levels = c("Variance", "LASSO", "PCA"))

  setnames(dt.depr, c("model", "feature_type"), c("Model", "Frequency"))

bin_plot <- ggplot(dt.depr[type == "bin"],
                   aes(x = test_accuracy, y = auc, color = Frequency, shape = Model)) +
    geom_point(size = 4, alpha = 0.8) +  # Larger points with slight transparency for better clarity
      theme_minimal(base_size = 14) +     # Minimal theme with larger base font size
      theme(
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  # Center and bold the title
        axis.title = element_text(size = 14, face = "bold"),              # Bold axis titles
        axis.text = element_text(size = 12),                              # Larger axis text
        legend.title = element_text(size = 13, face = "bold"),            # Bold legend titles
        legend.text = element_text(size = 11),                            # Larger legend text
        strip.text = element_text(size = 13, face = "bold")               # Bold facet labels
      ) +
      scale_color_brewer(palette = "Set1") +  # Use a visually appealing color palette
      scale_shape_manual(values = c(16, 17, 18, 19)) +  # Assign consistent shapes
      labs(
        x = "Model Accuracy (%)",
        y = "Area Under Curve (AUC)",
        color = "Adjacency Metric",
        shape = "Model Type"
      ) +
      facet_grid(. ~ optim)

weight_plot <- ggplot(dt.depr[type == "weight"],
                      aes(x = test_accuracy, y = auc, color = Frequency, shape = Model)) +
    geom_point(size = 4, alpha = 0.8) +  # Larger points with slight transparency for better clarity
      theme_minimal(base_size = 14) +     # Minimal theme with larger base font size
      theme(
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  # Center and bold the title
        axis.title = element_text(size = 14, face = "bold"),              # Bold axis titles
        axis.text = element_text(size = 12),                              # Larger axis text
        legend.title = element_text(size = 13, face = "bold"),            # Bold legend titles
        legend.text = element_text(size = 11),                            # Larger legend text
        strip.text = element_text(size = 13, face = "bold")               # Bold facet labels
      ) +
      scale_color_brewer(palette = "Set1") +  # Use a visually appealing color palette
      scale_shape_manual(values = c(16, 17, 18, 19)) +  # Assign consistent shapes
      labs(
        x = "Model Accuracy (%)",
        y = "Area Under Curve (AUC)",
        color = "Adjacency Metric",
        shape = "Model Type"
      ) +
      facet_grid(. ~ optim)

  res <- list("binarized" = bin_plot, "weighted" = weight_plot)

  return(res)
}

PlotDataAll <- function(dir) {

  dir_lasso <- paste0(dir, "ml_optim_output_lasso_combined_all/")

  dt.depr <- GetDataAll(dir_lasso, "lasso")

  dir_pca <- paste0(dir, "ml_optim_output_pca_combined_all/")

  dt.depr.tmp <- GetDataAll(dir_pca, "pca")

  if ("Var" %in% dt.depr.tmp$optim) {
  dt.depr.tmp <- dt.depr.tmp[optim != "Var"]
  }

  dt.depr <- rbind(dt.depr, dt.depr.tmp)

  dt.depr[model == "SVC", model := "SVM"]
  dt.depr[model == "RandomForestClassifier", model := "Random Forest"]
  dt.depr[model == "XGBClassifier", model := "XGBoost"]

  dt.depr[optim == "lasso", optim := "LASSO"]
  dt.depr[optim == "pca", optim := "PCA"]
  dt.depr[optim == "Var", optim := "Variance"]

  dt.depr[, test_accuracy := test_accuracy*100]
  dt.depr[, auc := auc*100]

  dt.depr$optim <- factor(dt.depr$optim, levels = c("Variance", "LASSO", "PCA"))

  setnames(dt.depr, c("model"), c("Model"))

bin_plot <- ggplot(dt.depr[type == "bin"],
                   aes(x = test_accuracy, y = auc, shape = Model)) +
    geom_point(size = 4, alpha = 0.8) +  # Larger points with slight transparency for better clarity
      theme_minimal(base_size = 14) +     # Minimal theme with larger base font size
      theme(
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  # Center and bold the title
        axis.title = element_text(size = 14, face = "bold"),              # Bold axis titles
        axis.text = element_text(size = 12),                              # Larger axis text
        legend.title = element_text(size = 13, face = "bold"),            # Bold legend titles
        legend.text = element_text(size = 11),                            # Larger legend text
        strip.text = element_text(size = 13, face = "bold")               # Bold facet labels
      ) +
      scale_color_brewer(palette = "Set1") +  # Use a visually appealing color palette
      scale_shape_manual(values = c(16, 17, 18, 19)) +  # Assign consistent shapes
      labs(
        x = "Model Accuracy (%)",
        y = "Area Under Curve (AUC)",
        shape = "Model Type"
      ) +
      facet_grid(. ~ optim)

weight_plot <- ggplot(dt.depr[type == "weight"],
                      aes(x = test_accuracy, y = auc, shape = Model)) +
    geom_point(size = 4, alpha = 0.8) +  # Larger points with slight transparency for better clarity
      theme_minimal(base_size = 14) +     # Minimal theme with larger base font size
      theme(
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  # Center and bold the title
        axis.title = element_text(size = 14, face = "bold"),              # Bold axis titles
        axis.text = element_text(size = 12),                              # Larger axis text
        legend.title = element_text(size = 13, face = "bold"),            # Bold legend titles
        legend.text = element_text(size = 11),                            # Larger legend text
        strip.text = element_text(size = 13, face = "bold")               # Bold facet labels
      ) +
      scale_color_brewer(palette = "Set1") +  # Use a visually appealing color palette
      scale_shape_manual(values = c(16, 17, 18, 19)) +  # Assign consistent shapes
      labs(
        x = "Model Accuracy (%)",
        y = "Area Under Curve (AUC)",
        shape = "Model Type"
      ) +
      facet_grid(. ~ optim)

  res <- list("binarized" = bin_plot, "weighted" = weight_plot)

  return(res)
}

GetInference <- function(dt, y_col, feat_vec) {


  dt_res <- data.table()

  for (feat in feat_vec) {
    # Perform Welch's ANOVA for the feature
    kruskal_result <- kruskal.test(get(y_col) ~ get(feat), data = dt)

    # Calculate omega squared and its confidence interval
    eta <- rstatix::kruskal_effsize(get(y_col) ~ get(feat), data = dt)

    # Create a temporary data table to store the results for each feature
    dt_res_tmp <- data.table(
      "feature" = paste(feat),
      "p_value" = kruskal_result$p.value,
      "eta_2_eff_size" = eta$effsize,
      "eta_2_magn" = eta$magnitude
    )

    # Append the temporary results to the main result data table
    dt_res <- rbind(dt_res, dt_res_tmp)
  }

  return(dt_res)
}

PlotDiffEffectSizes <- function(datasets) {
  dt_res_non_agg <- data.table()

  for (dataset in datasets) {
    dir_depr <- paste0("H:/magistro_studijos/magis/final_results_3/", dataset,"/")
    dir_lasso <- paste0(dir_depr, "ml_optim_output_lasso/")
    dt.depr <- GetData(dir_lasso, "lasso")
    dir_pca <- paste0(dir_depr, "ml_optim_output_pca/")
    dt.depr.pca <- GetData(dir_pca, "pca")
    dt.depr <- rbind(dt.depr, dt.depr.pca)
    dt.depr <- dt.depr[, .(type, optim, freq, feature_type, model, test_accuracy, auc)]

    # Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
    inf_res_acc <- GetInference(dt.depr, "test_accuracy", c("freq", "feature_type", "type", "optim", "model"))

    # Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
    inf_res_auc <-GetInference(dt.depr, "auc", c("freq", "feature_type", "type", "optim", "model"))

    dt_res <- rbind(
      inf_res_acc[, target := "Accuracy"],
      inf_res_auc[, target := "AUC"]
    )

    dt_res[, data := dataset]

    dt_res_non_agg <- rbind(dt_res_non_agg, dt_res)

    rm(dt.depr)
  }

  dt_res_non_agg[, agg := "Non aggregated"]

  dt_res_agg_bands <- data.table()

  for (dataset in datasets) {
    dir_depr <- paste0("H:/magistro_studijos/magis/final_results_3/", dataset,"/")
    dir_lasso <- paste0(dir_depr, "ml_optim_output_lasso_combined_bands/")
    dt.depr <- GetDataBands(dir_lasso, "lasso")
    dir_pca <- paste0(dir_depr, "ml_optim_output_pca_combined_bands/")
    dt.depr.pca <- GetDataBands(dir_pca, "pca")
    dt.depr <- rbind(dt.depr, dt.depr.pca)
    dt.depr <- dt.depr[, .(type, optim, feature_type, model, test_accuracy, auc)]

    # Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
    inf_res_acc <- GetInference(dt.depr, "test_accuracy", c("feature_type", "type", "optim", "model"))

    # Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
    inf_res_auc <-GetInference(dt.depr, "auc", c("feature_type", "type", "optim", "model"))

    dt_res <- rbind(
      inf_res_acc[, target := "Accuracy"],
      inf_res_auc[, target := "AUC"]
    )

    dt_res[, data := dataset]

    dt_res_agg_bands <- rbind(dt_res_agg_bands, dt_res)

    rm(dt.depr)
  }

  dt_res_agg_bands[, agg := "Aggregated on bands"]

  dt_res_agg_metrics <- data.table()

  for (dataset in datasets) {
    dir_depr <- paste0("H:/magistro_studijos/magis/final_results_3/", dataset,"/")
    dir_lasso <- paste0(dir_depr, "ml_optim_output_lasso_combined_metrics/")
    dt.depr <- GetDataMetrics(dir_lasso, "lasso")
    dir_pca <- paste0(dir_depr, "ml_optim_output_pca_combined_metrics/")
    dt.depr.pca <- GetDataMetrics(dir_pca, "pca")
    dt.depr <- rbind(dt.depr, dt.depr.pca)
    dt.depr <- dt.depr[, .(type, optim, feature_type, model, test_accuracy, auc)]

    if (anyNA(dt.depr)) dt.depr <- na.omit(dt.depr)

    # Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
    inf_res_acc <- GetInference(dt.depr, "test_accuracy", c("feature_type", "type", "optim", "model"))

    # Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
    inf_res_auc <-GetInference(dt.depr, "auc", c("feature_type", "type", "optim", "model"))

    dt_res <- rbind(
      inf_res_acc[, target := "Accuracy"],
      inf_res_auc[, target := "AUC"]
    )

    dt_res[, data := dataset]

    dt_res_agg_metrics <- rbind(dt_res_agg_metrics, dt_res)

    rm(dt.depr)
  }

  dt_res_agg_metrics[, agg := "Aggregated on metrics"]

  dt_res_agg_all <- data.table()

  for (dataset in datasets) {
    dir_depr <- paste0("H:/magistro_studijos/magis/final_results_3/", dataset,"/")
    dir_lasso <- paste0(dir_depr, "ml_optim_output_lasso_combined_all/")
    dt.depr <- GetDataAll(dir_lasso, "lasso")
    dir_pca <- paste0(dir_depr, "ml_optim_output_pca_combined_all/")
    dt.depr.pca <- GetDataAll(dir_pca, "pca")
    dt.depr <- rbind(dt.depr, dt.depr.pca)
    dt.depr <- dt.depr[, .(type, optim, model, test_accuracy, auc)]

    if (anyNA(dt.depr)) dt.depr <- na.omit(dt.depr)

    # Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
    inf_res_acc <- GetInference(dt.depr, "test_accuracy", c("type", "optim", "model"))

    # Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
    inf_res_auc <-GetInference(dt.depr, "auc", c("type", "optim", "model"))

    dt_res <- rbind(
      inf_res_acc[, target := "Accuracy"],
      inf_res_auc[, target := "AUC"]
    )

    dt_res[, data := dataset]

    dt_res_agg_all <- rbind(dt_res_agg_all, dt_res)

    rm(dt.depr)
  }


  dt_res_agg_all[, agg := "Aggregated fully"]

  dt_res_plot <- rbind(dt_res_non_agg, dt_res_agg_bands, dt_res_agg_metrics, dt_res_agg_all)

  # add significance flag
  dt_res_plot[p_value >= 0.05, Significance := "p-value >= 0.05"]
  dt_res_plot[p_value < 0.05, Significance := "p-value < 0.05"]

  # rename features and add order
  dt_res_plot[feature == "freq", feature := "Frequency selection"]
  dt_res_plot[feature == "feature_type", feature := "Feature extraction"]
  dt_res_plot[feature == "Feature extraction" & agg == "Aggregated on metrics", feature := "Frequency selection"]
  dt_res_plot[feature == "type", feature := "Binarization"]
  dt_res_plot[feature == "optim", feature := "Feature selection"]
  dt_res_plot[feature == "model", feature := "Model type"]

  dt_res_plot[data == "data_depression", data := "RVPH"]
  dt_res_plot[data == "data_eyes", data := "MMI"]

  dt_res_plot$feature <- factor(dt_res_plot$feature, levels = c("Frequency selection", "Feature extraction",
                                                                "Binarization", "Feature selection", "Model type"))

  dt_res_plot$agg <- factor(dt_res_plot$agg, levels = c("Non aggregated", "Aggregated on metrics",
                                                        "Aggregated on bands","Aggregated fully"))

  dt_res_plot$data <- factor(dt_res_plot$data, levels = c("RVPH", "MMI"))

  plot_res <- ggplot(dt_res_plot, aes(x = feature, y = eta_2_eff_size, color = data, group = interaction(data, target, agg)))+
              geom_line(size = 1.2, alpha = 0.8) +  # Thicker lines with slight transparency
              geom_point(aes(shape = Significance), size = 4) +  # Larger points for clarity
              facet_grid(target ~ agg, scales = "free") +  # Free scales to adjust per facet
              theme_minimal(base_size = 14) +  # Minimal theme with larger base font size
              theme(
                plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  # Center and bold the title
                axis.text.x = element_text(angle = 45, hjust = 1, size = 12),  # Rotate, resize, and align x-axis labels
                axis.text.y = element_text(size = 12),  # Resize y-axis labels
                axis.title = element_text(size = 14, face = "bold"),  # Bold axis titles
                legend.title = element_text(size = 13, face = "bold"),  # Bold legend title
                legend.text = element_text(size = 11),  # Larger legend text
                strip.text = element_text(size = 13, face = "bold")  # Bold facet labels
              ) +
              scale_color_brewer(palette = "Set2") +  # Use a professional color palette
              scale_shape_manual(values = c(16, 17, 18, 19)) +  # Standardized shapes
              labs(
                x = "Feature",
                y = expression(eta^2),  # Use LaTeX-like math expression for eta-squared
                color = "Dataset",
                shape = "Significance"
              ) +
              guides(color = guide_legend(override.aes = list(size = 4)))

  res <- list("plot" = plot_res,
              "table" = dt_res_plot)

  return(res)
}

PlotDiffRFWorkflow <- function(datasets) {

  dt_res_non_agg <- data.table()

  for (dataset in datasets) {
    dir_depr <- paste0("H:/magistro_studijos/magis/final_results_3/", dataset,"/")
    dir_lasso <- paste0(dir_depr, "ml_optim_output_lasso/")
    dt.depr <- GetData(dir_lasso, "lasso")
    dir_pca <- paste0(dir_depr, "ml_optim_output_pca/")
    dt.depr.pca <- GetData(dir_pca, "pca")
    dt.depr <- rbind(dt.depr, dt.depr.pca)
    dt.depr <- dt.depr[, .(type, optim, freq, feature_type, model, test_accuracy, auc)]

    # Feature importance - with optim == "Var"
    rf_model_acc <- randomForest(test_accuracy ~ ., data = dt.depr[, !c("auc")], mtry = 5)
    importance_values_acc <- as.data.frame(importance(rf_model_acc))
    importance_values_acc <- importance_values_acc %>%
    mutate(
      feature = rownames(importance_values_acc),
      share = IncNodePurity / sum(IncNodePurity),
      target = "Accuracy",
      data = dataset
    )

    rf_model_auc <- randomForest(auc ~ ., data = dt.depr[, !c("test_accuracy")], mtry = 5)
    importance_values_auc <- as.data.frame(importance(rf_model_auc))
    importance_values_auc %>% mutate(share = IncNodePurity / sum(importance_values_auc$IncNodePurity))
    importance_values_auc <- importance_values_auc %>%
    mutate(
      feature = rownames(importance_values_auc),
      share = IncNodePurity / sum(IncNodePurity),
      target = "AUC",
      data = dataset
    )

    dt_res <- rbind(
      as.data.table(importance_values_acc),
      as.data.table(importance_values_auc)
    )

    dt_res_non_agg <- rbind(dt_res_non_agg, dt_res)

    rm(dt.depr)
  }

  dt_res_non_agg[, agg := "Non aggregated"]

  dt_res_agg_bands <- data.table()

  for (dataset in datasets) {
    dir_depr <- paste0("H:/magistro_studijos/magis/final_results_3/", dataset,"/")
    dir_lasso <- paste0(dir_depr, "ml_optim_output_lasso_combined_bands/")
    dt.depr <- GetDataBands(dir_lasso, "lasso")
    dir_pca <- paste0(dir_depr, "ml_optim_output_pca_combined_bands/")
    dt.depr.pca <- GetDataBands(dir_pca, "pca")
    dt.depr <- rbind(dt.depr, dt.depr.pca)
    dt.depr <- dt.depr[, .(type, optim, feature_type, model, test_accuracy, auc)]

    # Feature importance - with optim == "Var"
    rf_model_acc <- randomForest(test_accuracy ~ ., data = dt.depr[, !c("auc")], mtry = 4)
    importance_values_acc <- as.data.frame(importance(rf_model_acc))
    importance_values_acc <- importance_values_acc %>%
    mutate(
      feature = rownames(importance_values_acc),
      share = IncNodePurity / sum(IncNodePurity),
      target = "Accuracy",
      data = dataset
    )

    rf_model_auc <- randomForest(auc ~ ., data = dt.depr[, !c("test_accuracy")], mtry = 4)
    importance_values_auc <- as.data.frame(importance(rf_model_auc))
    importance_values_auc %>% mutate(share = IncNodePurity / sum(importance_values_auc$IncNodePurity))
    importance_values_auc <- importance_values_auc %>%
    mutate(
      feature = rownames(importance_values_auc),
      share = IncNodePurity / sum(IncNodePurity),
      target = "AUC",
      data = dataset
    )

    dt_res <- rbind(
      as.data.table(importance_values_acc),
      as.data.table(importance_values_auc)
    )

    dt_res_agg_bands <- rbind(dt_res_agg_bands, dt_res)

    rm(dt.depr)
  }

  dt_res_agg_bands[, agg := "Aggregated on bands"]

  dt_res_agg_metrics <- data.table()

  for (dataset in datasets) {
    dir_depr <- paste0("H:/magistro_studijos/magis/final_results_3/", dataset,"/")
    dir_lasso <- paste0(dir_depr, "ml_optim_output_lasso_combined_metrics/")
    dt.depr <- GetDataMetrics(dir_lasso, "lasso")
    dir_pca <- paste0(dir_depr, "ml_optim_output_pca_combined_metrics/")
    dt.depr.pca <- GetDataMetrics(dir_pca, "pca")
    dt.depr <- rbind(dt.depr, dt.depr.pca)
    dt.depr <- dt.depr[, .(type, optim, feature_type, model, test_accuracy, auc)]

    if (anyNA(dt.depr)) dt.depr <- na.omit(dt.depr)

    # Feature importance - with optim == "Var"
    rf_model_acc <- randomForest(test_accuracy ~ ., data = dt.depr[, !c("auc")], mtry = 4)
    importance_values_acc <- as.data.frame(importance(rf_model_acc))
    importance_values_acc <- importance_values_acc %>%
    mutate(
      feature = rownames(importance_values_acc),
      share = IncNodePurity / sum(IncNodePurity),
      target = "Accuracy",
      data = dataset
    )

    rf_model_auc <- randomForest(auc ~ ., data = dt.depr[, !c("test_accuracy")], mtry = 4)
    importance_values_auc <- as.data.frame(importance(rf_model_auc))
    importance_values_auc %>% mutate(share = IncNodePurity / sum(importance_values_auc$IncNodePurity))
    importance_values_auc <- importance_values_auc %>%
    mutate(
      feature = rownames(importance_values_auc),
      share = IncNodePurity / sum(IncNodePurity),
      target = "AUC",
      data = dataset
    )

    dt_res <- rbind(
      as.data.table(importance_values_acc),
      as.data.table(importance_values_auc)
    )

    dt_res_agg_metrics <- rbind(dt_res_agg_metrics, dt_res)

    rm(dt.depr)
  }

  dt_res_agg_metrics[, agg := "Aggregated on metrics"]

  dt_res_agg_all <- data.table()

  for (dataset in datasets) {
    dir_depr <- paste0("H:/magistro_studijos/magis/final_results_3/", dataset,"/")
    dir_lasso <- paste0(dir_depr, "ml_optim_output_lasso_combined_all/")
    dt.depr <- GetDataAll(dir_lasso, "lasso")
    dir_pca <- paste0(dir_depr, "ml_optim_output_pca_combined_all/")
    dt.depr.pca <- GetDataAll(dir_pca, "pca")
    dt.depr <- rbind(dt.depr, dt.depr.pca)
    dt.depr <- dt.depr[, .(type, optim, model, test_accuracy, auc)]

    if (anyNA(dt.depr)) dt.depr <- na.omit(dt.depr)

    # Feature importance - with optim == "Var"
    rf_model_acc <- randomForest(test_accuracy ~ ., data = dt.depr[, !c("auc")], mtry = 3)
    importance_values_acc <- as.data.frame(importance(rf_model_acc))
    importance_values_acc <- importance_values_acc %>%
    mutate(
      feature = rownames(importance_values_acc),
      share = IncNodePurity / sum(IncNodePurity),
      target = "Accuracy",
      data = dataset
    )

    rf_model_auc <- randomForest(auc ~ ., data = dt.depr[, !c("test_accuracy")], mtry = 3)
    importance_values_auc <- as.data.frame(importance(rf_model_auc))
    importance_values_auc %>% mutate(share = IncNodePurity / sum(importance_values_auc$IncNodePurity))
    importance_values_auc <- importance_values_auc %>%
    mutate(
      feature = rownames(importance_values_auc),
      share = IncNodePurity / sum(IncNodePurity),
      target = "AUC",
      data = dataset
    )

    dt_res <- rbind(
      as.data.table(importance_values_acc),
      as.data.table(importance_values_auc)
    )

    dt_res_agg_all <- rbind(dt_res_agg_all, dt_res)

    rm(dt.depr)
  }

  dt_res_agg_all[, agg := "Aggregated fully"]

  dt_res_plot <- rbind(dt_res_non_agg, dt_res_agg_bands, dt_res_agg_metrics, dt_res_agg_all)

  # rename features and add order
  dt_res_plot[feature == "freq", feature := "Frequency selection"]
  dt_res_plot[feature == "feature_type", feature := "Feature extraction"]
  dt_res_plot[feature == "Feature extraction" & agg == "Aggregated on metrics", feature := "Frequency selection"]
  dt_res_plot[feature == "type", feature := "Binarization"]
  dt_res_plot[feature == "optim", feature := "Feature selection"]
  dt_res_plot[feature == "model", feature := "Model type"]

  dt_res_plot[data == "data_depression", data := "RVPH"]
  dt_res_plot[data == "data_eyes", data := "MMI"]

  dt_res_plot[, share := share * 100]

  dt_res_plot$feature <- factor(dt_res_plot$feature, levels = c("Frequency selection", "Feature extraction",
                                                                "Binarization", "Feature selection", "Model type"))

  dt_res_plot$agg <- factor(dt_res_plot$agg, levels = c("Non aggregated", "Aggregated on metrics",
                                                        "Aggregated on bands", "Aggregated fully"))

  dt_res_plot$data <- factor(dt_res_plot$data, levels = c("RVPH", "MMI"))

  plot_res <- ggplot(dt_res_plot, aes(x = feature, y = share, color = data, group = interaction(data, target, agg)))+
              geom_line(size = 1.2, alpha = 0.8) +  # Thicker lines with slight transparency
              facet_grid(target ~ agg, scales = "free") +  # Free scales to adjust per facet
              theme_minimal(base_size = 14) +  # Minimal theme with larger base font size
              theme(
                plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  # Center and bold the title
                axis.text.x = element_text(angle = 45, hjust = 1, size = 12),  # Rotate, resize, and align x-axis labels
                axis.text.y = element_text(size = 12),  # Resize y-axis labels
                axis.title = element_text(size = 14, face = "bold"),  # Bold axis titles
                legend.title = element_text(size = 13, face = "bold"),  # Bold legend title
                legend.text = element_text(size = 11),  # Larger legend text
                strip.text = element_text(size = 13, face = "bold")  # Bold facet labels
              ) +
              scale_color_brewer(palette = "Set2") +  # Use a professional color palette
              scale_shape_manual(values = c(16, 17, 18, 19)) +  # Standardized shapes
              labs(
                x = "Feature",
                y = "Importance share",  # Use LaTeX-like math expression for eta-squared
                color = "Dataset",
                shape = "Significance"
              ) +
              guides(color = guide_legend(override.aes = list(size = 4)))

  res <- list("plot" = plot_res,
            "table" = dt_res_plot)

  return(res)
}

PlotDiffRFWorkflowNoInt <- function(datasets) {

  dt_res_non_agg <- data.table()

  for (dataset in datasets) {
    dir_depr <- paste0("H:/magistro_studijos/magis/final_results_3/", dataset,"/")
    dir_lasso <- paste0(dir_depr, "ml_optim_output_lasso/")
    dt.depr <- GetData(dir_lasso, "lasso")
    dir_pca <- paste0(dir_depr, "ml_optim_output_pca/")
    dt.depr.pca <- GetData(dir_pca, "pca")
    dt.depr <- rbind(dt.depr, dt.depr.pca)
    dt.depr <- dt.depr[, .(type, optim, freq, feature_type, model, test_accuracy, auc)]

    # Feature importance - with optim == "Var"
    rf_model_acc <- randomForest(test_accuracy ~ ., data = dt.depr[, !c("auc")], mtry = 1)
    importance_values_acc <- as.data.frame(importance(rf_model_acc))
    importance_values_acc <- importance_values_acc %>%
    mutate(
      feature = rownames(importance_values_acc),
      share = IncNodePurity / sum(IncNodePurity),
      target = "Accuracy",
      data = dataset
    )

    rf_model_auc <- randomForest(auc ~ ., data = dt.depr[, !c("test_accuracy")], mtry = 1)
    importance_values_auc <- as.data.frame(importance(rf_model_auc))
    importance_values_auc %>% mutate(share = IncNodePurity / sum(importance_values_auc$IncNodePurity))
    importance_values_auc <- importance_values_auc %>%
    mutate(
      feature = rownames(importance_values_auc),
      share = IncNodePurity / sum(IncNodePurity),
      target = "AUC",
      data = dataset
    )

    dt_res <- rbind(
      as.data.table(importance_values_acc),
      as.data.table(importance_values_auc)
    )

    dt_res_non_agg <- rbind(dt_res_non_agg, dt_res)

    rm(dt.depr)
  }

  dt_res_non_agg[, agg := "Non aggregated"]

  dt_res_agg_bands <- data.table()

  for (dataset in datasets) {
    dir_depr <- paste0("H:/magistro_studijos/magis/final_results_3/", dataset,"/")
    dir_lasso <- paste0(dir_depr, "ml_optim_output_lasso_combined_bands/")
    dt.depr <- GetDataBands(dir_lasso, "lasso")
    dir_pca <- paste0(dir_depr, "ml_optim_output_pca_combined_bands/")
    dt.depr.pca <- GetDataBands(dir_pca, "pca")
    dt.depr <- rbind(dt.depr, dt.depr.pca)
    dt.depr <- dt.depr[, .(type, optim, feature_type, model, test_accuracy, auc)]

    # Feature importance - with optim == "Var"
    rf_model_acc <- randomForest(test_accuracy ~ ., data = dt.depr[, !c("auc")], mtry = 1)
    importance_values_acc <- as.data.frame(importance(rf_model_acc))
    importance_values_acc <- importance_values_acc %>%
    mutate(
      feature = rownames(importance_values_acc),
      share = IncNodePurity / sum(IncNodePurity),
      target = "Accuracy",
      data = dataset
    )

    rf_model_auc <- randomForest(auc ~ ., data = dt.depr[, !c("test_accuracy")], mtry = 1)
    importance_values_auc <- as.data.frame(importance(rf_model_auc))
    importance_values_auc %>% mutate(share = IncNodePurity / sum(importance_values_auc$IncNodePurity))
    importance_values_auc <- importance_values_auc %>%
    mutate(
      feature = rownames(importance_values_auc),
      share = IncNodePurity / sum(IncNodePurity),
      target = "AUC",
      data = dataset
    )

    dt_res <- rbind(
      as.data.table(importance_values_acc),
      as.data.table(importance_values_auc)
    )

    dt_res_agg_bands <- rbind(dt_res_agg_bands, dt_res)

    rm(dt.depr)
  }

  dt_res_agg_bands[, agg := "Aggregated on bands"]

  dt_res_agg_metrics <- data.table()

  for (dataset in datasets) {
    dir_depr <- paste0("H:/magistro_studijos/magis/final_results_3/", dataset,"/")
    dir_lasso <- paste0(dir_depr, "ml_optim_output_lasso_combined_metrics/")
    dt.depr <- GetDataMetrics(dir_lasso, "lasso")
    dir_pca <- paste0(dir_depr, "ml_optim_output_pca_combined_metrics/")
    dt.depr.pca <- GetDataMetrics(dir_pca, "pca")
    dt.depr <- rbind(dt.depr, dt.depr.pca)
    dt.depr <- dt.depr[, .(type, optim, feature_type, model, test_accuracy, auc)]

    if (anyNA(dt.depr)) dt.depr <- na.omit(dt.depr)

    # Feature importance - with optim == "Var"
    rf_model_acc <- randomForest(test_accuracy ~ ., data = dt.depr[, !c("auc")], mtry = 1)
    importance_values_acc <- as.data.frame(importance(rf_model_acc))
    importance_values_acc <- importance_values_acc %>%
    mutate(
      feature = rownames(importance_values_acc),
      share = IncNodePurity / sum(IncNodePurity),
      target = "Accuracy",
      data = dataset
    )

    rf_model_auc <- randomForest(auc ~ ., data = dt.depr[, !c("test_accuracy")], mtry = 1)
    importance_values_auc <- as.data.frame(importance(rf_model_auc))
    importance_values_auc %>% mutate(share = IncNodePurity / sum(importance_values_auc$IncNodePurity))
    importance_values_auc <- importance_values_auc %>%
    mutate(
      feature = rownames(importance_values_auc),
      share = IncNodePurity / sum(IncNodePurity),
      target = "AUC",
      data = dataset
    )

    dt_res <- rbind(
      as.data.table(importance_values_acc),
      as.data.table(importance_values_auc)
    )

    dt_res_agg_metrics <- rbind(dt_res_agg_metrics, dt_res)

    rm(dt.depr)
  }

  dt_res_agg_metrics[, agg := "Aggregated on metrics"]

  dt_res_agg_all <- data.table()

  for (dataset in datasets) {
    dir_depr <- paste0("H:/magistro_studijos/magis/final_results_3/", dataset,"/")
    dir_lasso <- paste0(dir_depr, "ml_optim_output_lasso_combined_all/")
    dt.depr <- GetDataAll(dir_lasso, "lasso")
    dir_pca <- paste0(dir_depr, "ml_optim_output_pca_combined_all/")
    dt.depr.pca <- GetDataAll(dir_pca, "pca")
    dt.depr <- rbind(dt.depr, dt.depr.pca)
    dt.depr <- dt.depr[, .(type, optim, model, test_accuracy, auc)]

    if (anyNA(dt.depr)) dt.depr <- na.omit(dt.depr)

    # Feature importance - with optim == "Var"
    rf_model_acc <- randomForest(test_accuracy ~ ., data = dt.depr[, !c("auc")], mtry = 1)
    importance_values_acc <- as.data.frame(importance(rf_model_acc))
    importance_values_acc <- importance_values_acc %>%
    mutate(
      feature = rownames(importance_values_acc),
      share = IncNodePurity / sum(IncNodePurity),
      target = "Accuracy",
      data = dataset
    )

    rf_model_auc <- randomForest(auc ~ ., data = dt.depr[, !c("test_accuracy")], mtry = 1)
    importance_values_auc <- as.data.frame(importance(rf_model_auc))
    importance_values_auc %>% mutate(share = IncNodePurity / sum(importance_values_auc$IncNodePurity))
    importance_values_auc <- importance_values_auc %>%
    mutate(
      feature = rownames(importance_values_auc),
      share = IncNodePurity / sum(IncNodePurity),
      target = "AUC",
      data = dataset
    )

    dt_res <- rbind(
      as.data.table(importance_values_acc),
      as.data.table(importance_values_auc)
    )

    dt_res_agg_all <- rbind(dt_res_agg_all, dt_res)

    rm(dt.depr)
  }

  dt_res_agg_all[, agg := "Aggregated fully"]

  dt_res_plot <- rbind(dt_res_non_agg, dt_res_agg_bands, dt_res_agg_metrics, dt_res_agg_all)


  # rename features and add order
  dt_res_plot[feature == "freq", feature := "Frequency selection"]
  dt_res_plot[feature == "feature_type", feature := "Feature extraction"]
  dt_res_plot[feature == "Feature extraction" & agg == "Aggregated on metrics", feature := "Frequency selection"]
  dt_res_plot[feature == "type", feature := "Binarization"]
  dt_res_plot[feature == "optim", feature := "Feature selection"]
  dt_res_plot[feature == "model", feature := "Model type"]

  dt_res_plot[data == "data_depression", data := "RVPH"]
  dt_res_plot[data == "data_eyes", data := "MMI"]

  dt_res_plot[, share := share * 100]

  dt_res_plot$feature <- factor(dt_res_plot$feature, levels = c("Frequency selection", "Feature extraction",
                                                                "Binarization", "Feature selection", "Model type"))

  dt_res_plot$agg <- factor(dt_res_plot$agg, levels = c("Non aggregated", "Aggregated on metrics",
                                                        "Aggregated on bands","Aggregated fully"))

  dt_res_plot$data <- factor(dt_res_plot$data, levels = c("RVPH", "MMI"))

  plot_res <- ggplot(dt_res_plot, aes(x = feature, y = share, color = data, group = interaction(data, target, agg)))+
              geom_line(size = 1.2, alpha = 0.8) +  # Thicker lines with slight transparency
              facet_grid(target ~ agg, scales = "free") +  # Free scales to adjust per facet
              theme_minimal(base_size = 14) +  # Minimal theme with larger base font size
              theme(
                plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  # Center and bold the title
                axis.text.x = element_text(angle = 45, hjust = 1, size = 12),  # Rotate, resize, and align x-axis labels
                axis.text.y = element_text(size = 12),  # Resize y-axis labels
                axis.title = element_text(size = 14, face = "bold"),  # Bold axis titles
                legend.title = element_text(size = 13, face = "bold"),  # Bold legend title
                legend.text = element_text(size = 11),  # Larger legend text
                strip.text = element_text(size = 13, face = "bold")  # Bold facet labels
              ) +
              scale_color_brewer(palette = "Set2") +  # Use a professional color palette
              scale_shape_manual(values = c(16, 17, 18, 19)) +  # Standardized shapes
              labs(
                x = "Feature",
                y = "Importance share",  # Use LaTeX-like math expression for eta-squared
                color = "Dataset",
                shape = "Significance"
              ) +
              guides(color = guide_legend(override.aes = list(size = 4)))

  res <- list("plot" = plot_res,
              "table" = dt_res_plot)

  return(res)
}

ConvertToPlotly <- function(ls) {
  res_ls <- list()  # Initialize an empty list to store plotly plots

  # Loop through each ggplot object in the list
  for(i in seq_along(ls)) {
    res_ls[[i]] <- ggplotly(ls[[i]])  # Convert each ggplot object to plotly and store it
  }

  return(res_ls)  # Return the list of plotly plots
}

#----Top performing models----

top_depr <- GetDatasetResults("data_depression")

top_5_depr <- top_depr[order(-test_accuracy, -auc)][1:5]

top_eyes <- GetDatasetResults("data_eyes")

top_5_eyes <- top_eyes[order(-test_accuracy, -auc)][1:5]

#----Plots----

# comparison of feature importances

dir <- "H:/magistro_studijos/magis/final_results_3/"

eff_plot_comp <- PlotDiffEffectSizes(c("data_depression", "data_eyes"))

eff_sizes_res <- as.data.table(eff_plot_comp$table)

ggsave(paste0(dir, "figures/eff_plot_comp.png"), plot = eff_plot_comp$plot, width = 8, height = 6, dpi = 300)

set.seed(123)

imp_plot_comp <- PlotDiffRFWorkflow(c("data_depression", "data_eyes"))

imp_res <- as.data.table(imp_plot_comp$table)

ggsave(paste0(dir, "figures/imp_plot_comp.png"), plot = imp_plot_comp$plot, width = 8, height = 6, dpi = 300)


imp_plot_comp_no_int <- PlotDiffRFWorkflowNoInt(c("data_depression", "data_eyes"))

imp_no_int_res <- as.data.table(imp_plot_comp_no_int$table)

ggsave(paste0(dir, "figures/imp_plot_comp_no_int.png"), plot = imp_plot_comp_no_int$plot, width = 8, height = 6, dpi = 300)
# data depression

dir_depr <- "H:/magistro_studijos/magis/final_results_3/data_depression/"

# data depression - non combined
rez_depr <- PlotData(dir_depr)
# rez_depr_CV <- PlotDataCV(dir_depr)

ggsave(paste0(dir_depr, "figures/rvph_non_agg_binarized_all.png"), plot = rez_depr$binarized_all, width = 8, height = 3, dpi = 300)
ggsave(paste0(dir_depr, "figures/rvph_non_agg_binarized_freq.png"), plot = rez_depr$binarized_freq, width = 8, height = 9, dpi = 300)
ggsave(paste0(dir_depr, "figures/rvph_non_agg_weighted_all.png"), plot = rez_depr$weighted_all, width = 8, height = 3, dpi = 300)
ggsave(paste0(dir_depr, "figures/rvph_non_agg_weighted_freq.png"), plot = rez_depr$weighted_freq, width = 8, height = 9, dpi = 300)

# convert to plotly

rez_depr_plotly <- ConvertToPlotly(rez_depr)


# data depression - combined on bands
rez_depr_bands <- PlotDataBands(dir_depr)

ggsave(paste0(dir_depr, "figures/rvph_agg_bands_binarized.png"), plot = rez_depr_bands$binarized, width = 8, height = 3, dpi = 300)
ggsave(paste0(dir_depr, "figures/rvph_agg_bands_weighted.png"), plot = rez_depr_bands$weighted, width = 8, height = 3, dpi = 300)

# convert to plotly
rez_depr_bands_plotly <- ConvertToPlotly(rez_depr_bands)

# data depression - combined on metrics
rez_depr_metrics <- PlotDataMetrics(dir_depr)

ggsave(paste0(dir_depr, "figures/rvph_agg_metrics_binarized.png"), plot = rez_depr_metrics$binarized, width = 8, height = 3, dpi = 300)
ggsave(paste0(dir_depr, "figures/rvph_agg_metrics_weighted.png"), plot = rez_depr_metrics$weighted, width = 8, height = 3, dpi = 300)

# convert to plotly
rez_depr_metrics_plotly <- ConvertToPlotly(rez_depr_metrics)


# data depression - combined all

rez_depr_all <- PlotDataAll(dir_depr)

ggsave(paste0(dir_depr, "figures/rvph_agg_all_binarized.png"), plot = rez_depr_all$binarized, width = 8, height = 3, dpi = 300)
ggsave(paste0(dir_depr, "figures/rvph_agg_all_weighted.png"), plot = rez_depr_all$weighted, width = 8, height = 3, dpi = 300)

# convert to plotly
rez_depr_all_plotly <- ConvertToPlotly(rez_depr_all)







# data eyes
dir_eyes <- "H:/magistro_studijos/magis/final_results_3/data_eyes/"

# data eyes - non combined
rez_eyes <- PlotData(dir_eyes)

ggsave(paste0(dir_eyes, "figures/mmi_non_agg_binarized_all.png"), plot = rez_eyes$binarized_all, width = 8, height = 3, dpi = 300)
ggsave(paste0(dir_eyes, "figures/mmi_non_agg_binarized_freq.png"), plot = rez_eyes$binarized_freq, width = 8, height = 9, dpi = 300)
ggsave(paste0(dir_eyes, "figures/mmi_non_agg_weighted_all.png"), plot = rez_eyes$weighted_all, width = 8, height = 3, dpi = 300)
ggsave(paste0(dir_eyes, "figures/mmi_non_agg_weighted_freq.png"), plot = rez_eyes$weighted_freq, width = 8, height = 9, dpi = 300)

# convert to plotly
rez_eyes_plotly <- ConvertToPlotly(rez_eyes)

# data eyes - combined on bands
rez_eyes_bands <- PlotDataBands(dir_eyes)

ggsave(paste0(dir_eyes, "figures/mmi_agg_bands_binarized.png"), plot = rez_eyes_bands$binarized, width = 8, height = 3, dpi = 300)
ggsave(paste0(dir_eyes, "figures/mmi_agg_bands_weighted.png"), plot = rez_eyes_bands$weighted, width = 8, height = 3, dpi = 300)

rez_eyes_bands_plotly <- ConvertToPlotly(rez_eyes_bands)

# data eyes - combined on metrics
rez_eyes_metrics <- PlotDataMetrics(dir_eyes)

ggsave(paste0(dir_eyes, "figures/mmi_agg_metrics_binarized.png"), plot = rez_eyes_metrics$binarized, width = 8, height = 3, dpi = 300)
ggsave(paste0(dir_eyes, "figures/mmi_agg_metrics_weighted.png"), plot = rez_eyes_metrics$weighted, width = 8, height = 3, dpi = 300)

rez_eyes_metrics_plotly <- ConvertToPlotly(rez_eyes_metrics)

# data eyes - combined all

rez_eyes_all <- PlotDataAll(dir_eyes)

ggsave(paste0(dir_eyes, "figures/mmi_agg_all_binarized.png"), plot = rez_eyes_all$binarized, width = 8, height = 3, dpi = 300)
ggsave(paste0(dir_eyes, "figures/mmi_agg_all_weighted.png"), plot = rez_eyes_all$weighted, width = 8, height = 3, dpi = 300)

# convert to plotly
rez_eyes_all_plotly <- ConvertToPlotly(rez_eyes_all)

#----Tests aggregation type----

top_depr <- GetDatasetResults("data_depression")

# normality assumption is met, homogeneity of variance not met
# anova with interaction - shows what steps has significance
# Levene's Test for Homogeneity of Variance
levene_result <- leveneTest(test_accuracy ~ agg, data = top_depr)
print(levene_result)
anova_result_acc <- aov(test_accuracy ~ agg, data = top_depr)
residuals_anova <- residuals(anova_result_acc)
shapiro.test(residuals_anova)
# Q-Q Plot
qqnorm(residuals_anova)
qqline(residuals_anova, col = "red", lwd = 2)
# Levene's Test for Homogeneity of Variance
levene_result <- leveneTest(auc ~ agg, data = top_depr)
print(levene_result)
anova_result_auc <- aov(auc ~ agg, data = top_depr)
residuals_anova <- residuals(anova_result_auc)
shapiro.test(residuals_anova)
# Q-Q Plot
qqnorm(residuals_anova)
qqline(residuals_anova, col = "red", lwd = 2)

# test for all selection types
GetInference(top_depr, "test_accuracy", c("agg"))
pairwise.wilcox.test(top_depr$test_accuracy, top_depr$agg, p.adjust.method = "bonferroni") %>% suppressWarnings()

GetInference(top_depr, "auc", c("agg"))
pairwise.wilcox.test(top_depr$auc, top_depr$agg, p.adjust.method = "bonferroni") %>% suppressWarnings()



# Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
# for PCA
GetInference(top_depr[optim == "pca"], "test_accuracy", c("agg"))
pairwise.wilcox.test(top_depr[optim == "pca"]$test_accuracy, top_depr[optim == "pca"]$agg, p.adjust.method = "bonferroni") %>% suppressWarnings()

# for variance
GetInference(top_depr[optim == "Var"], "test_accuracy", c("agg"))
pairwise.wilcox.test(top_depr[optim == "Var"]$test_accuracy, top_depr[optim == "Var"]$agg, p.adjust.method = "bonferroni") %>% suppressWarnings()

# for lasso
GetInference(top_depr[optim == "lasso"], "test_accuracy", c("agg"))
pairwise.wilcox.test(top_depr[optim == "lasso"]$test_accuracy, top_depr[optim == "lasso"]$agg, p.adjust.method = "bonferroni") %>% suppressWarnings()

# Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
# for PCA
GetInference(top_depr[optim == "pca"], "auc", c("agg"))
pairwise.wilcox.test(top_depr[optim == "pca"]$auc, top_depr[optim == "pca"]$agg, p.adjust.method = "bonferroni") %>% suppressWarnings()

# for variance
GetInference(top_depr[optim == "var"], "auc", c("agg"))
pairwise.wilcox.test(top_depr[optim == "var"]$auc, top_depr[optim == "var"]$agg, p.adjust.method = "bonferroni") %>% suppressWarnings()

# for lasso
GetInference(top_depr[optim == "lasso"], "auc", c("agg"))
pairwise.wilcox.test(top_depr[optim == "lasso"]$auc, top_depr[optim == "lasso"]$agg, p.adjust.method = "bonferroni") %>% suppressWarnings()




top_eyes <- GetDatasetResults("data_eyes")

# normality assumption is met, homogeneity of variance not met
# anova with interaction - shows what steps has significance
# Levene's Test for Homogeneity of Variance
levene_result <- leveneTest(test_accuracy ~ agg, data = top_eyes)
print(levene_result)
anova_result_acc <- aov(test_accuracy ~ agg, data = top_eyes)
residuals_anova <- residuals(anova_result_acc)
shapiro.test(residuals_anova)
# Q-Q Plot
qqnorm(residuals_anova)
qqline(residuals_anova, col = "red", lwd = 2)
# Levene's Test for Homogeneity of Variance
levene_result <- leveneTest(auc ~ agg, data = top_eyes)
print(levene_result)
anova_result_auc <- aov(auc ~ agg, data = top_eyes)
residuals_anova <- residuals(anova_result_auc)
shapiro.test(residuals_anova)
# Q-Q Plot
qqnorm(residuals_anova)
qqline(residuals_anova, col = "red", lwd = 2)


# test for all selection types
GetInference(top_eyes, "test_accuracy", c("agg"))
pairwise.wilcox.test(top_eyes$test_accuracy, top_eyes$agg, p.adjust.method = "bonferroni") %>% suppressWarnings()

GetInference(top_eyes, "auc", c("agg"))
pairwise.wilcox.test(top_eyes$auc, top_eyes$agg, p.adjust.method = "bonferroni") %>% suppressWarnings()


# Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
# for PCA
GetInference(top_eyes[optim == "pca"], "test_accuracy", c("agg"))
pairwise.wilcox.test(top_eyes[optim == "pca"]$test_accuracy, top_eyes[optim == "pca"]$agg, p.adjust.method = "bonferroni") %>% suppressWarnings()

# for variance
GetInference(top_eyes[optim == "var"], "test_accuracy", c("agg"))
pairwise.wilcox.test(top_eyes[optim == "var"]$test_accuracy, top_eyes[optim == "var"]$agg, p.adjust.method = "bonferroni") %>% suppressWarnings()

# for lasso
GetInference(top_eyes[optim == "lasso"], "test_accuracy", c("agg"))
pairwise.wilcox.test(top_eyes[optim == "lasso"]$test_accuracy, top_eyes[optim == "lasso"]$agg, p.adjust.method = "bonferroni") %>% suppressWarnings()

# Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
# for PCA
GetInference(top_eyes[optim == "pca"], "auc", c("agg"))
pairwise.wilcox.test(top_eyes[optim == "pca"]$auc, top_eyes[optim == "pca"]$agg, p.adjust.method = "bonferroni") %>% suppressWarnings()

# for variance
GetInference(top_eyes[optim == "var"], "auc", c("agg"))
pairwise.wilcox.test(top_eyes[optim == "var"]$auc, top_eyes[optim == "var"]$agg, p.adjust.method = "bonferroni") %>% suppressWarnings()

# for lasso
GetInference(top_eyes[optim == "lasso"], "auc", c("agg"))
pairwise.wilcox.test(top_eyes[optim == "lasso"]$auc, top_eyes[optim == "lasso"]$agg, p.adjust.method = "bonferroni") %>% suppressWarnings()

#----Tests on meta features----

# analysis of results - depression

dir_depr <- "H:/magistro_studijos/magis/final_results_3/data_depression/"



dir_lasso <- paste0(dir_depr, "ml_optim_output_lasso/")

dt.depr <- GetData(dir_lasso, "lasso")

dir_pca <- paste0(dir_depr, "ml_optim_output_pca/")

dt.depr.pca <- GetData(dir_pca, "pca")

dt.depr <- rbind(dt.depr, dt.depr.pca)

dt.depr <- dt.depr[, .(type, optim, freq, feature_type, model, test_accuracy, auc)]


# Levene's Test for Homogeneity of Variance
levene_result <- leveneTest(test_accuracy ~ feature_type * optim * model * freq, data = dt.depr)
print(levene_result)

# anova with interaction - shows what steps has significance
anova_result_acc <- aov(test_accuracy ~ feature_type * optim * model * freq, data = dt.depr)
summary(anova_result_acc)

# Extract residuals
residuals_anova <- residuals(anova_result_acc)
shapiro.test(residuals_anova)

# Q-Q Plot
qqnorm(residuals_anova)
qqline(residuals_anova, col = "red", lwd = 2)

# Levene's Test for Homogeneity of Variance
levene_result <- leveneTest(auc ~ feature_type * optim * model * freq, data = dt.depr)
print(levene_result)

anova_result_auc <- aov(auc ~ feature_type * optim * model * freq, data = dt.depr)
summary(anova_result_auc)

# Extract residuals
residuals_anova <- residuals(anova_result_auc)
shapiro.test(residuals_anova)

# Q-Q Plot
qqnorm(residuals_anova)
qqline(residuals_anova, col = "red", lwd = 2)

# Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
GetInference(dt.depr, "test_accuracy", c("freq", "feature_type", "type", "optim", "model"))

# Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
GetInference(dt.depr, "auc", c("freq", "feature_type", "type", "optim", "model"))

set.seed(123)

# Feature importance - with optim == "Var"
rf_model_acc <- randomForest(test_accuracy ~ ., data = dt.depr[, !c("auc")], mtry = 5)
importance_values_acc <- as.data.frame(importance(rf_model_acc))
importance_values_acc %>% mutate(share = IncNodePurity / sum(importance_values_acc$IncNodePurity))

rf_model_auc <- randomForest(auc ~ ., data = dt.depr[, !c("test_accuracy")], mtry = 5)
importance_values_auc <- as.data.frame(importance(rf_model_auc))
importance_values_auc %>% mutate(share = IncNodePurity / sum(importance_values_auc$IncNodePurity))

# descriptive stats
dt.depr.max.acc <- dt.depr[, .SD[which.max(test_accuracy)], .(type, optim, freq)]
dt.depr.max.auc <- dt.depr[, .SD[which.max(auc)], .(type, optim, freq)]

dt.depr.acc.cnt <- dt.depr.max.acc[, .N, .(feature_type, model)][order(-N)]
dt.depr.auc.cnt <- dt.depr.max.auc[, .N, .(feature_type, model)][order(-N)]



# analysis of results - agg on bands

dir_lasso <- paste0(dir_depr, "ml_optim_output_lasso_combined_bands/")

dt.depr.bands <- GetDataBands(dir_lasso, "lasso")

dir_pca <- paste0(dir_depr, "ml_optim_output_pca_combined_bands/")

dt.depr.bands.pca <- GetDataBands(dir_pca, "pca")

dt.depr.bands <- rbind(dt.depr.bands, dt.depr.bands.pca)

dt.depr.bands <- dt.depr.bands[, .(type, optim, feature_type, model, test_accuracy, auc)]

# Levene's Test for Homogeneity of Variance
levene_result <- leveneTest(test_accuracy ~ feature_type * optim * model, data = dt.depr.bands)
print(levene_result)

# anova with interaction - shows what steps has significance
anova_result_acc <- aov(test_accuracy ~ feature_type * optim * model, data = dt.depr.bands)
summary(anova_result_acc)

# Extract residuals
residuals_anova <- residuals(anova_result_acc)
shapiro.test(residuals_anova)

# Q-Q Plot
qqnorm(residuals_anova)
qqline(residuals_anova, col = "red", lwd = 2)

# Levene's Test for Homogeneity of Variance
levene_result <- leveneTest(auc ~ feature_type * optim * model, data = dt.depr.bands)
print(levene_result)

anova_result_auc <- aov(auc ~ feature_type * optim * model, data = dt.depr.bands)
summary(anova_result_auc)

# Extract residuals
residuals_anova <- residuals(anova_result_auc)
shapiro.test(residuals_anova)

# Q-Q Plot
qqnorm(residuals_anova)
qqline(residuals_anova, col = "red", lwd = 2)


# Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
GetInference(dt.depr.bands, "test_accuracy", c("feature_type", "type", "optim", "model"))


# Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
GetInference(dt.depr.bands, "auc", c("feature_type", "type", "optim", "model"))

set.seed(123)

# Feature importance - with optim == "Var"
rf_model_acc <- randomForest(test_accuracy ~ ., data = dt.depr.bands[, !c("auc")], mtry=4)
importance_values_acc <- as.data.frame(importance(rf_model_acc))
importance_values_acc %>% mutate(share = IncNodePurity / sum(importance_values_acc$IncNodePurity))


rf_model_auc <- randomForest(auc ~ ., data = dt.depr.bands[, !c("test_accuracy")], mtry=4)
importance_values_auc <- as.data.frame(importance(rf_model_auc))
importance_values_auc %>% mutate(share = IncNodePurity / sum(importance_values_auc$IncNodePurity))

# descriptive stats
dt.depr.max.acc <- dt.depr.bands[, .SD[which.max(test_accuracy)], .(type, optim)]
dt.depr.max.auc <- dt.depr.bands[, .SD[which.max(auc)], .(type, optim)]

dt.depr.acc.cnt <- dt.depr.max.acc[, .N, .(feature_type, model)][order(-N)]
dt.depr.auc.cnt <- dt.depr.max.auc[, .N, .(feature_type, model)][order(-N)]



# analysis of results - agg on metrics (feature type as in frequency!)

dir_lasso <- paste0(dir_depr, "ml_optim_output_lasso_combined_metrics/")

dt.depr.metrics <- GetDataMetrics(dir_lasso, "lasso")

dir_pca <- paste0(dir_depr, "ml_optim_output_pca_combined_metrics/")

dt.depr.metrics.pca <- GetDataMetrics(dir_pca, "pca")

dt.depr.metrics <- rbind(dt.depr.metrics, dt.depr.metrics.pca)

dt.depr.metrics <- dt.depr.metrics[, .(type, optim, feature_type, model, test_accuracy, auc)]

# Levene's Test for Homogeneity of Variance
levene_result <- leveneTest(test_accuracy ~ feature_type * optim * model, data = dt.depr.metrics)
print(levene_result)

# anova with interaction - shows what steps has significance
anova_result_acc <- aov(test_accuracy ~ feature_type * optim * model, data = dt.depr.metrics)
summary(anova_result_acc)

# Extract residuals
residuals_anova <- residuals(anova_result_acc)
shapiro.test(residuals_anova)

# Q-Q Plot
qqnorm(residuals_anova)
qqline(residuals_anova, col = "red", lwd = 2)

# Levene's Test for Homogeneity of Variance
levene_result <- leveneTest(auc ~ feature_type * optim * model, data = dt.depr.metrics)
print(levene_result)

anova_result_auc <- aov(auc ~ feature_type * optim * model, data = dt.depr.metrics)
summary(anova_result_auc)

# Extract residuals
residuals_anova <- residuals(anova_result_auc)
shapiro.test(residuals_anova)

# Q-Q Plot
qqnorm(residuals_anova)
qqline(residuals_anova, col = "red", lwd = 2)


# Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
GetInference(dt.depr.metrics, "test_accuracy", c("feature_type", "type", "optim", "model"))

# Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
GetInference(dt.depr.metrics, "auc", c("feature_type", "type", "optim", "model"))

set.seed(123)

# Feature importance - with optim == "Var"
rf_model_acc <- randomForest(test_accuracy ~ ., data = dt.depr.metrics[, !c("auc")], mtry = 4)
importance_values_acc <- as.data.frame(importance(rf_model_acc))
importance_values_acc %>% mutate(share = IncNodePurity / sum(importance_values_acc$IncNodePurity))

rf_model_auc <- randomForest(auc ~ ., data = dt.depr.metrics[, !c("test_accuracy")], mtry = 4)
importance_values_auc <- as.data.frame(importance(rf_model_auc))
importance_values_auc %>% mutate(share = IncNodePurity / sum(importance_values_auc$IncNodePurity))

# descriptive stats
dt.depr.metrics.max.acc <- dt.depr.metrics[, .SD[which.max(test_accuracy)], .(type, optim)]
dt.depr.metrics.max.auc <- dt.depr.metrics[, .SD[which.max(auc)], .(type, optim)]

dt.depr.metrics.acc.cnt <- dt.depr.metrics.max.acc[, .N, .(feature_type, model)][order(-N)]
dt.depr.metrics.auc.cnt <- dt.depr.metrics.max.auc[, .N, .(feature_type, model)][order(-N)]






# analysis of results eyes

dir_eyes <- "H:/magistro_studijos/magis/final_results_3/data_eyes/"




dir_lasso <- paste0(dir_eyes, "ml_optim_output_lasso/")

dt.eyes <- GetData(dir_lasso, "lasso")

dir_pca <- paste0(dir_eyes, "ml_optim_output_pca/")

dt.eyes.pca <- GetData(dir_pca, "pca")

dt.eyes <- rbind(dt.eyes, dt.eyes.pca)

dt.eyes <- dt.eyes[, .(type, optim, freq, feature_type, model, test_accuracy, auc)]

# Levene's Test for Homogeneity of Variance
levene_result <- leveneTest(test_accuracy ~ feature_type * optim * model * freq, data = dt.eyes)
print(levene_result)

# anova with interaction - shows what steps has significance
anova_result_acc <- aov(test_accuracy ~ feature_type * optim * model * freq, data = dt.eyes)
summary(anova_result_acc)

# Extract residuals
residuals_anova <- residuals(anova_result_acc)
shapiro.test(residuals_anova)

# Q-Q Plot
qqnorm(residuals_anova)
qqline(residuals_anova, col = "red", lwd = 2)

# Levene's Test for Homogeneity of Variance
levene_result <- leveneTest(auc ~ feature_type * optim * model * freq, data = dt.eyes)
print(levene_result)

anova_result_auc <- aov(auc ~ feature_type * optim * model * freq, data = dt.eyes)
summary(anova_result_auc)

# Extract residuals
residuals_anova <- residuals(anova_result_auc)
shapiro.test(residuals_anova)

# Q-Q Plot
qqnorm(residuals_anova)
qqline(residuals_anova, col = "red", lwd = 2)


# Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
GetInference(dt.eyes, "test_accuracy", c("freq", "feature_type", "type", "optim", "model"))

# Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
GetInference(dt.eyes, "auc", c("freq", "feature_type", "type", "optim", "model"))

set.seed(123)

# Feature importance - with optim == "Var"
rf_model_acc <- randomForest(test_accuracy ~ ., data = dt.eyes[, !c("auc")], mtry = 5)
importance_values_acc <- as.data.frame(importance(rf_model_acc))
importance_values_acc %>% mutate(share = IncNodePurity / sum(importance_values_acc$IncNodePurity))

rf_model_auc <- randomForest(auc ~ ., data = dt.eyes[, !c("test_accuracy")], mtry = 5)
importance_values_auc <- as.data.frame(importance(rf_model_auc))
importance_values_auc %>% mutate(share = IncNodePurity / sum(importance_values_auc$IncNodePurity))

# descriptive stats
dt.eyes.max.acc <- dt.eyes[, .SD[which.max(test_accuracy)], .(type, optim, freq)]
dt.eyes.max.auc <- dt.eyes[, .SD[which.max(auc)], .(type, optim, freq)]

dt.eyes.acc.cnt <- dt.eyes.max.acc[, .N, .(feature_type, model)][order(-N)]
dt.eyes.auc.cnt <- dt.eyes.max.auc[, .N, .(feature_type, model)][order(-N)]



# analysis of results - agg on bands

dir_lasso <- paste0(dir_eyes, "ml_optim_output_lasso_combined_bands/")

dt.eyes.bands <- GetDataBands(dir_lasso, "lasso")

dir_pca <- paste0(dir_eyes, "ml_optim_output_pca_combined_bands/")

dt.eyes.bands.pca <- GetDataBands(dir_pca, "pca")

dt.eyes.bands <- rbind(dt.eyes.bands, dt.eyes.bands.pca)

dt.eyes.bands <- dt.eyes.bands[, .(type, optim, feature_type, model, test_accuracy, auc)]

# Levene's Test for Homogeneity of Variance
levene_result <- leveneTest(test_accuracy ~ feature_type * optim * model, data = dt.eyes.bands)
print(levene_result)

# anova with interaction - shows what steps has significance
anova_result_acc <- aov(test_accuracy ~ feature_type * optim * model, data = dt.eyes.bands)
summary(anova_result_acc)

# Extract residuals
residuals_anova <- residuals(anova_result_acc)
shapiro.test(residuals_anova)

# Q-Q Plot
qqnorm(residuals_anova)
qqline(residuals_anova, col = "red", lwd = 2)

# Levene's Test for Homogeneity of Variance
levene_result <- leveneTest(auc ~ feature_type * optim * model, data = dt.eyes.bands)
print(levene_result)

anova_result_auc <- aov(auc ~ feature_type * optim * model, data = dt.eyes.bands)
summary(anova_result_auc)

# Extract residuals
residuals_anova <- residuals(anova_result_auc)
shapiro.test(residuals_anova)

# Q-Q Plot
qqnorm(residuals_anova)
qqline(residuals_anova, col = "red", lwd = 2)

# Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
GetInference(dt.eyes.bands, "test_accuracy", c("feature_type", "type", "optim", "model"))

# Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
GetInference(dt.eyes.bands, "auc", c("feature_type", "type", "optim", "model"))


set.seed(123)

# Feature importance - with optim == "Var"
rf_model_acc <- randomForest(test_accuracy ~ ., data = dt.eyes.bands[, !c("auc")], mtry = 4)
importance_values_acc <- as.data.frame(importance(rf_model_acc))
importance_values_acc %>% mutate(share = IncNodePurity / sum(importance_values_acc$IncNodePurity))

rf_model_auc <- randomForest(auc ~ ., data = dt.eyes.bands[, !c("test_accuracy")], mtry = 4)
importance_values_auc <- as.data.frame(importance(rf_model_auc))
importance_values_auc %>% mutate(share = IncNodePurity / sum(importance_values_auc$IncNodePurity))

# descriptive stats
dt.eyes.bands.max.acc <- dt.eyes.bands[, .SD[which.max(test_accuracy)], .(type, optim)]
dt.eyes.bands.max.auc <- dt.eyes.bands[, .SD[which.max(auc)], .(type, optim)]

dt.eyes.bands.acc.cnt <- dt.eyes.bands.max.acc[, .N, .(feature_type, model)][order(-N)]
dt.eyes.bands.auc.cnt <- dt.eyes.bands.max.auc[, .N, .(feature_type, model)][order(-N)]


# analysis of results - agg on metrics (feature type as in frequency!)

dir_lasso <- paste0(dir_eyes, "ml_optim_output_lasso_combined_metrics/")

dt.eyes.metrics <- GetDataMetrics(dir_lasso, "lasso")

dir_pca <- paste0(dir_eyes, "ml_optim_output_pca_combined_metrics/")

dt.eyes.metrics.pca <- GetDataMetrics(dir_pca, "pca")

dt.eyes.metrics <- rbind(dt.eyes.metrics, dt.eyes.metrics.pca)

dt.eyes.metrics <- dt.eyes.metrics[, .(type, optim, feature_type, model, test_accuracy, auc)]


# Levene's Test for Homogeneity of Variance
levene_result <- leveneTest(test_accuracy ~ feature_type * optim * model, data = dt.eyes.metrics)
print(levene_result)

# anova with interaction - shows what steps has significance
anova_result_acc <- aov(test_accuracy ~ feature_type * optim * model, data = dt.eyes.metrics)
summary(anova_result_acc)

# Extract residuals
residuals_anova <- residuals(anova_result_acc)
shapiro.test(residuals_anova)

# Q-Q Plot
qqnorm(residuals_anova)
qqline(residuals_anova, col = "red", lwd = 2)

# Levene's Test for Homogeneity of Variance
levene_result <- leveneTest(auc ~ feature_type * optim * model, data = dt.eyes.metrics)
print(levene_result)

anova_result_auc <- aov(auc ~ feature_type * optim * model, data = dt.eyes.metrics)
summary(anova_result_auc)

# Extract residuals
residuals_anova <- residuals(anova_result_auc)
shapiro.test(residuals_anova)

# Q-Q Plot
qqnorm(residuals_anova)
qqline(residuals_anova, col = "red", lwd = 2)


# Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
GetInference(dt.eyes.metrics, "test_accuracy", c("feature_type", "type", "optim", "model"))

# Perform Welch’s ANOVA for the main effect of feature_type (for a one-way design)
GetInference(dt.eyes.metrics, "auc", c("feature_type", "type", "optim", "model"))


set.seed(123)

# Feature importance - with optim == "Var"
rf_model_acc <- randomForest(test_accuracy ~ ., data = dt.eyes.metrics[, !c("auc")], mtry = 4)
importance_values_acc <- as.data.frame(importance(rf_model_acc))
importance_values_acc %>% mutate(share = IncNodePurity / sum(importance_values_acc$IncNodePurity))

rf_model_auc <- randomForest(auc ~ ., data = dt.eyes.metrics[, !c("test_accuracy")], mtry = 4)
importance_values_auc <- as.data.frame(importance(rf_model_auc))
importance_values_auc %>% mutate(share = IncNodePurity / sum(importance_values_auc$IncNodePurity))

# descriptive stats
dt.eyes.metrics.max.acc <- dt.eyes.metrics[, .SD[which.max(test_accuracy)], .(type, optim)]
dt.eyes.metrics.max.auc <- dt.eyes.metrics[, .SD[which.max(auc)], .(type, optim)]

dt.eyes.metrics.acc.cnt <- dt.eyes.metrics.max.acc[, .N, .(feature_type, model)][order(-N)]
dt.eyes.metrics.auc.cnt <- dt.eyes.metrics.max.auc[, .N, .(feature_type, model)][order(-N)]



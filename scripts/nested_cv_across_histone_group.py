from IPython.display import clear_output
import random
from tqdm import tqdm
import joblib
import uuid 

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet, ARDRegression

def nested_cross_validation_across_histone_group(adata, n_folds=5, hyperparams=[0.05, 0.01], histone_group=['H3K4me3', 'H3K4me1']):
    """
    Perform nested cross-validation across a given histone group.

    Arguments:
        adata: AnnData object containing expression data and metadata.
        n_folds: int, Number of folds for cross-validation.
        hyperparams: list, Hyperparameters for the ElasticNet model.
        histone_group: list, Histone group to be considered.

    Returns:
        adata: Updated AnnData object with cross-validation results.
    """

    # Extract data from the adata object
    X = adata.X
    y = adata.obs['age']

    # Initialize data frames to store validation, test predictions, and standard deviations
    adata.obsm['cv_val_df'] = pd.DataFrame(columns=['cv_val' + str(i+1) for i in range(n_folds)], index=adata.obs_names, dtype=float)
    adata.obsm['cv_test_df'] = pd.DataFrame(columns=['cv_test' + str(i+1) for i in range(n_folds)], index=adata.obs_names, dtype=float)
    adata.obsm['cv_test_std_df'] = pd.DataFrame(columns=['cv_test_std' + str(i+1) for i in range(n_folds)], index=adata.obs_names, dtype=float)

    # Divide folds by biosample, making sure replicates are together to not inflate score
    biosample_accessions = np.unique(adata.obs.biosample_accession)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    kf.get_n_splits(biosample_accessions)
    folds_experiment = [list(fold) for _, fold in kf.split(biosample_accessions)]
    folds = []
    for fold in folds_experiment:
        biosample_accessions_fold = biosample_accessions[fold]
        fold = list(np.where([experiment_accession in biosample_accessions_fold for experiment_accession in adata.obs.biosample_accession])[0])
        folds += [fold]

    # Main loop for nested cross-validation
    for fold_number, test_fold in enumerate(folds):

        # Preparing train and test sets
        folds_without_test = [item for item in folds if item != test_fold]
        train_fold = [item for sublist in folds_without_test for item in sublist]

        # Filter histones
        X_train = X[train_fold][[histone in histone_group for histone in adata.obs.histone.iloc[train_fold].tolist()]]
        y_train = y[train_fold][[histone in histone_group for histone in adata.obs.histone.iloc[train_fold].tolist()]]
        X_test = X[test_fold]
        y_test = y[test_fold]

        errors = []
        for hyperparam in hyperparams:

            y_hat_val = []
            # Loop for hyperparameter tuning
            for val_test_fold in folds_without_test:
                # Preparing validation train and test sets
                folds_without_val_test = [n for n in folds_without_test if n != val_test_fold]
                val_train_fold = [item for sublist in folds_without_val_test for item in sublist]
                X_val_train = X[val_train_fold][[histone in histone_group for histone in adata.obs.histone.iloc[val_train_fold].tolist()]]
                y_val_train = y[val_train_fold][[histone in histone_group for histone in adata.obs.histone.iloc[val_train_fold].tolist()]]
                X_val_test = X[val_test_fold][[histone in histone_group for histone in adata.obs.histone.iloc[val_test_fold].tolist()]]
                y_val_test = y[val_test_fold][[histone in histone_group for histone in adata.obs.histone.iloc[val_test_fold].tolist()]]

                # Feature selection and dimensionality reduction
                feature_selector = ElasticNet(alpha=hyperparam, max_iter=1000, l1_ratio=0.9, random_state=42)
                feature_selector.fit(X_val_train, y_val_train)
                features = (np.abs(feature_selector.coef_) > 0)
                dim_reduction = TruncatedSVD(n_components=np.min([len(X_val_train)-1, np.sum(features)-1]))
                X_val_train = dim_reduction.fit_transform(X_val_train[:, features])
                X_val_test = dim_reduction.transform(X_val_test[:, features])

                # Model training and prediction
                model = ARDRegression(n_iter=1000)
                model.fit(X_val_train, y_val_train)
                y_hat_val_test = model.predict(X_val_test)
                y_hat_val += list(y_hat_val_test)

                # Storing validation predictions
                adata.obsm['cv_val_df']['cv_val' + str(fold_number+1)][np.intersect1d(val_test_fold, np.where([histone in histone_group for histone in adata.obs.histone.tolist()])[0])] = y_hat_val_test

            # Compute error for each hyperparameter
            errors += [np.mean(np.abs(y_train - y_hat_val)**2)]

        # Select best hyperparameter based on errors
        best_hyperparam = hyperparams[np.where(errors == np.min(errors))[0][0]]

        # Repeat the feature selection, dimensionality reduction, and model training with the best hyperparameter for the test data
        feature_selector = ElasticNet(alpha=best_hyperparam, max_iter=1000, l1_ratio=0.9, random_state=42)
        feature_selector.fit(X_train, y_train)
        features = (np.abs(feature_selector.coef_) > 0)
        dim_reduction = TruncatedSVD(n_components=np.min([len(X_train)-1, np.sum(features)-1]))
        X_train = dim_reduction.fit_transform(X_train[:, features])
        X_test = dim_reduction.transform(X_test[:, features])
        model = ARDRegression(n_iter=1000)
        model.fit(X_train, y_train)
        y_hat_test, y_hat_std_test = model.predict(X_test, return_std=True)

        # Store the paths and models for further use
        feature_selector_path = 'results/adata_dependencies/' + uuid.uuid4().hex[:20]
        adata.uns['cv_test'+str(fold_number)+'_feature_selector'] = feature_selector_path
        joblib.dump(feature_selector, feature_selector_path)
        dim_reduction_path = 'results/adata_dependencies/' + uuid.uuid4().hex[:20]
        adata.uns['cv_test'+str(fold_number)+'_dim_reduction'] = dim_reduction_path
        joblib.dump(dim_reduction, dim_reduction_path)
        model_path = 'results/adata_dependencies/' + uuid.uuid4().hex[:20]
        adata.uns['cv_test'+str(fold_number)+'_model'] = model_path
        joblib.dump(model, model_path)

        # Store test predictions and standard deviations
        adata.obsm['cv_test_df']['cv_test' + str(fold_number+1)][test_fold] = y_hat_test
        adata.obsm['cv_test_std_df']['cv_test_std' + str(fold_number+1)][test_fold] = y_hat_std_test

        # Clear the output to keep the console tidy
        clear_output()

    return adata

# Read the adata_genes file which contains gene data
adata_genes = sc.read_h5ad('data/encode_data/adata_genes.h5ad')

# Subset the observations in the adata_genes object by excluding those marked as cancer
adata_genes._inplace_subset_obs([not x for x in adata_genes.obs['cancer']])

# Perform nested cross-validation across histones on the adata_genes object
adata_genes_across_activating_histone_group = nested_cross_validation_across_histone_group(
    adata=adata_genes, 
    n_folds=10,
    hyperparams=[0.1, 0.05, 0.01, 0.005, 0.001],
    histone_group=['H3K4me3', 'H3K27ac', 'H3K9ac'],
)

# Write the results of the nested cross-validation to an h5ad file
adata_genes_across_activating_histone_group.write_h5ad('results/adata_genes_across_activating_histone_group.h5ad')

adata_genes_across_repressing_histone_group = nested_cross_validation_across_histone_group(
    adata=adata_genes, 
    n_folds=10,
    hyperparams=[0.1, 0.05, 0.01, 0.005, 0.001],
    histone_group=['H3K9me3', 'H3K27me3', 'H3K36me3'],
)
adata_genes_across_repressing_histone_group.write_h5ad('results/adata_genes_across_repressing_histone_group.h5ad')

# Read the adata_genes file which contains gene expression data
adata_genes = sc.read_h5ad('data/encode_data/adata_genes.h5ad')

# Subset the observations in the adata_genes object by excluding those marked as cancer
adata_genes._inplace_subset_obs([not x for x in adata_genes.obs['cancer']])

# Perform nested cross-validation across histones on the adata_genes object
adata_genes_across_histones_flipped = nested_cross_validation_across_histones_flipped(
    adata=adata_genes, 
    n_folds=10,
    hyperparams=[0.1, 0.05, 0.01, 0.005, 0.001],
)
                  
# Write the results of the nested cross-validation to an h5ad file
adata_genes_across_histones_flipped.write_h5ad('results/adata_genes_across_histones_flipped.h5ad')
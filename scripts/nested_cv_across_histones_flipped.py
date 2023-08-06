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

def nested_cross_validation_across_histones_flipped(adata, n_folds=5, hyperparams=[0.05, 0.01]):
    """
    Performs nested cross-validation on data across different histones.
    The outer loop is responsible for testing, and the inner loop is responsible for validation.
    
    Arguments:
    adata: AnnData object containing the data matrix 'X' and observations 'obs'.
    n_folds: int, number of folds for cross-validation.
    hyperparams: list, hyperparameters to be tuned.
    
    Outputs:
    adata: Modified AnnData object including cross-validation results.
    """

    # Extracting data and target variable
    X = adata.X
    y = adata.obs['age']

    # Iterate over unique histones
    for i in tqdm(range(len(np.unique(adata.obs['histone']))):

        histone = np.unique(adata.obs['histone'])[i]

        # Initialize data frames to store cross-validation results for the current histone
        adata.obsm[histone + '_cv_val_df'] = pd.DataFrame(columns=['cv_val' + str(i+1) for i in range(n_folds)], index=adata.obs_names, dtype=float)
        adata.obsm[histone + '_cv_test_df'] = pd.DataFrame(columns=['cv_test' + str(i+1) for i in range(n_folds)], index=adata.obs_names, dtype=float)
        adata.obsm[histone + '_cv_test_std_df'] = pd.DataFrame(columns=['cv_test_std' + str(i+1) for i in range(n_folds)], index=adata.obs_names, dtype=float)

        # Divide folds by biosample, ensuring replicates are together to not inflate the score
        biosample_accessions = np.unique(adata.obs.biosample_accession)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        kf.get_n_splits(biosample_accessions)
        folds_experiment = [list(fold) for _, fold in kf.split(biosample_accessions)]
        folds = []
        for fold in folds_experiment:
            biosample_accessions_fold = biosample_accessions[fold]
            fold = list(np.where([experiment_accession in biosample_accessions_fold for experiment_accession in adata.obs.biosample_accession])[0])
            folds += [fold]

        # Outer loop for each fold (testing)
        for fold_number, test_fold in enumerate(folds):

            folds_without_test = [item for item in folds if item != test_fold]
            train_fold = [item for sublist in folds_without_test for item in sublist]

            # Splitting data for training and testing
            X_train = X[train_fold][adata.obs.histone.iloc[train_fold] == histone]
            y_train = y[train_fold][adata.obs.histone.iloc[train_fold] == histone]
            X_test = X[test_fold]
            y_test = y[test_fold]

            errors = []

            # Inner loop for hyperparameter tuning (validation)
            for hyperparam in hyperparams:

                y_hat_val = []
                for val_test_fold in folds_without_test:

                    folds_without_val_test = [n for n in folds_without_test if n != val_test_fold]
                    val_train_fold = [item for sublist in folds_without_val_test for item in sublist]

                    # Splitting data for training and validation
                    X_val_train = X[val_train_fold][adata.obs.histone.iloc[val_train_fold] == histone]
                    y_val_train = y[val_train_fold][adata.obs.histone.iloc[val_train_fold] == histone]
                    X_val_test = X[val_test_fold][adata.obs.histone.iloc[val_test_fold] == histone]
                    y_val_test = y[val_test_fold][adata.obs.histone.iloc[val_test_fold] == histone]

                    # Feature selection using ElasticNet
                    feature_selector = ElasticNet(alpha=hyperparam, max_iter=1000, l1_ratio=0.9, random_state=42)
                    feature_selector.fit(X_val_train, y_val_train)

                    features = (np.abs(feature_selector.coef_) > 0)

                    # Dimensionality reduction using TruncatedSVD
                    dim_reduction = TruncatedSVD(n_components=np.min([len(X_val_train)-1, np.sum(features)-1]))
                    X_val_train = dim_reduction.fit_transform(X_val_train[:,features])
                    X_val_test = dim_reduction.transform(X_val_test[:,features])

                    # Model training using ARDRegression
                    model = ARDRegression(n_iter=1000)
                    model.fit(X_val_train, y_val_train)

                    y_hat_val_test = model.predict(X_val_test)
                    y_hat_val += list(y_hat_val_test)

                    # Store validation predictions
                    adata.obsm[histone + '_cv_val_df']['cv_val' + str(fold_number+1)][np.intersect1d(val_test_fold, np.where(adata.obs.histone == histone)[0])] = y_hat_val_test

                # Compute mean squared error and store it
                errors += [np.mean(np.abs(y_train - y_hat_val)**2)]

            # Select the best hyperparameter based on errors
            best_hyperparam = hyperparams[np.where(errors == np.min(errors))[0][0]]

            # Repeat the feature selection, dimensionality reduction, and model training with the best hyperparameter for the test data
            feature_selector = ElasticNet(alpha=best_hyperparam, max_iter=1000, l1_ratio=0.9, random_state=42)
            feature_selector.fit(X_train, y_train)
            features = (np.abs(feature_selector.coef_) > 0)
            dim_reduction = TruncatedSVD(n_components=np.min([len(X_train)-1, np.sum(features)-1]))
            X_train = dim_reduction.fit_transform(X_train[:,features])
            X_test = dim_reduction.transform(X_test[:,features])
            model = ARDRegression(n_iter=1000)
            model.fit(X_train, y_train)
            y_hat_test, y_hat_std_test = model.predict(X_test, return_std=True)

            # Store the paths and models for further use
            feature_selector_path = 'results/adata_dependencies/' + uuid.uuid4().hex[:20]
            adata.uns[histone + '_cv_test'+str(fold_number)+'_feature_selector'] = feature_selector_path
            joblib.dump(feature_selector, feature_selector_path)
            dim_reduction_path = 'results/adata_dependencies/' + uuid.uuid4().hex[:20]
            adata.uns[histone + '_cv_test'+str(fold_number)+'_dim_reduction'] = dim_reduction_path
            joblib.dump(dim_reduction, dim_reduction_path)
            model_path = 'results/adata_dependencies/' + uuid.uuid4().hex[:20]
            adata.uns[histone + '_cv_test'+str(fold_number)+'_model'] = model_path
            joblib.dump(model, model_path)

            # Store test predictions and standard deviations
            adata.obsm[histone + '_cv_test_df']['cv_test' + str(fold_number+1)][test_fold] = y_hat_test
            adata.obsm[histone + '_cv_test_std_df']['cv_test_std' + str(fold_number+1)][test_fold] = y_hat_std_test

        clear_output()

    return adata

# Read the adata_genes file which contains gene data
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
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

def nested_cross_validation_pan_histone(adata, n_folds=5, hyperparams=[0.05, 0.01]):
    """
    This function performs nested cross-validation on the given data, focusing on a pan-histone analysis.
    The cross-validation is performed over different hyperparameters, and the results are stored within
    the input AnnData object.

    Arguments:
    adata: AnnData object containing the data matrix 'X' and observations 'obs'.
    n_folds: int, number of folds for cross-validation.
    hyperparams: list, hyperparameters to be tuned.

    Outputs:
    adata: Modified AnnData object including cross-validation results.
    """
    
    # Initialize data frames to store cross-validation results
    adata.obsm['cv_val_df'] = pd.DataFrame(columns=['cv_val' + str(i+1) for i in range(n_folds)], index=adata.obs_names, dtype=float)
    adata.obsm['cv_test_df'] = pd.DataFrame(columns=['cv_test' + str(i+1) for i in range(n_folds)], index=adata.obs_names, dtype=float)
    adata.obsm['cv_test_std_df'] = pd.DataFrame(columns=['cv_test_std' + str(i+1) for i in range(n_folds)], index=adata.obs_names, dtype=float)

    # Extract features and target variable
    X = adata.X
    y = adata.obs['age']

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

    # Outer cross-validation loop
    for fold_number, test_fold in enumerate(folds):
        folds_without_test = [item for item in folds if item != test_fold]
        train_fold = [item for sublist in folds_without_test for item in sublist]
        X_train = X[train_fold]
        y_train = y[train_fold]
        X_test = X[test_fold]
        y_test = y[test_fold]

        # List to store the errors for different hyperparameters
        errors = []

        # Inner cross-validation loop for hyperparameter tuning
        for hyperparam in hyperparams:
            y_hat_val = []

            # Loop through validation folds
            for val_test_fold in folds_without_test:
                folds_without_val_test = [n for n in folds_without_test if n != val_test_fold]
                val_train_fold = [item for sublist in folds_without_val_test for item in sublist]
                X_val_train = X[val_train_fold]
                y_val_train = y[val_train_fold]
                X_val_test = X[val_test_fold]
                y_val_test = y[val_test_fold]
                
                # Feature selection and dimensionality reduction using ElasticNet
                feature_selector =  ElasticNet(alpha=hyperparam, max_iter=1000, l1_ratio=0.9, random_state=42)
                feature_selector.fit(X_val_train, y_val_train)
                features = (np.abs(feature_selector.coef_) > 0)
                dim_reduction = TruncatedSVD(n_components=np.min([len(X_val_train)-1, np.sum(features)-1]))
                X_val_train = dim_reduction.fit_transform(X_val_train[:,features])
                X_val_test = dim_reduction.transform(X_val_test[:,features])

                # Model training using ARDRegression
                model = ARDRegression(n_iter=1000)
                model.fit(X_val_train, y_val_train)
                y_hat_val_test = model.predict(X_val_test)
                y_hat_val += list(y_hat_val_test)
                adata.obsm['cv_val_df']['cv_val' + str(fold_number+1)][val_test_fold] = y_hat_val_test

            # Compute mean squared error and store it
            errors += [np.mean(np.abs(y_train - y_hat_val)**2)]

        # Select best hyperparameter and train the final model with it
        best_hyperparam = hyperparams[np.where(errors == np.min(errors))[0][0]]
        feature_selector =  ElasticNet(alpha=best_hyperparam, max_iter=1000, l1_ratio=0.9, random_state=42)
        feature_selector.fit(X_train, y_train)
        features = (np.abs(feature_selector.coef_) > 0)
        dim_reduction = TruncatedSVD(n_components=np.min([len(X_train)-1, np.sum(features)-1]))
        X_train = dim_reduction.fit_transform(X_train[:,features])
        X_test = dim_reduction.transform(X_test[:,features])
        model = ARDRegression(n_iter=1000)
        model.fit(X_train, y_train)
        y_hat_test, y_hat_std_test = model.predict(X_test, return_std=True)

        # Save models and results
        feature_selector_path = 'results/adata_dependencies/' + uuid.uuid4().hex[:20]
        adata.uns['cv_test'+str(fold_number)+'_feature_selector'] = feature_selector_path
        joblib.dump(feature_selector, feature_selector_path)

        dim_reduction_path = 'results/adata_dependencies/' + uuid.uuid4().hex[:20]
        adata.uns['cv_test'+str(fold_number)+'_dim_reduction'] = dim_reduction_path
        joblib.dump(dim_reduction, dim_reduction_path)

        model_path = 'results/adata_dependencies/' + uuid.uuid4().hex[:20]
        adata.uns['cv_test'+str(fold_number)+'_model'] = model_path
        joblib.dump(model, model_path)

        # Store test results
        adata.obsm['cv_test_df']['cv_test' + str(fold_number+1)][test_fold] = y_hat_test
        adata.obsm['cv_test_std_df']['cv_test_std' + str(fold_number+1)][test_fold] = y_hat_std_test

        # Clear output to keep the terminal clean
        clear_output()
            
    return adata

# Read the gene data from a file
adata_genes = sc.read_h5ad('data/encode_data/adata_genes.h5ad')

# Subset the observations to exclude the ones marked as 'cancer'
adata_genes._inplace_subset_obs([not x for x in adata_genes.obs['cancer']])

# Perform nested cross-validation specifically for pan-histone using the gene data
# with specified number of folds and hyperparameters
adata_genes_pan_histone = nested_cross_validation_pan_histone(
    adata=adata_genes, 
    n_folds=10,
    hyperparams=[0.1, 0.05, 0.01, 0.005, 0.001],
)

# Save the pan-histone cross-validation results to an H5AD file
adata_genes_pan_histone.write_h5ad('results/adata_genes_pan_histone.h5ad')

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

def nested_cross_validation(adata, n_folds=5, hyperparams=[0.05, 0.01]):
    """
    This function performs nested cross-validation. The process involves splitting the data into training,
    validation, and test sets to evaluate and tune the models with different hyperparameters. The models are fitted using
    ElasticNet for feature selection and ARDRegression for regression. The results are stored within the input AnnData object.

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

    # Extracting data and target variable
    X = adata.X
    y = adata.obs['age']

    # Loop through unique histones
    for i in tqdm(range(len(np.unique(adata.obs['histone'])))):
        histone = np.unique(adata.obs['histone'])[i]

        filters = (adata.obs['histone'] == histone)
        X_histone = X[filters]
        y_histone = y[filters]

        # Divide folds by biosample, ensuring replicates are together to not inflate the score
        biosample_accessions = np.unique(adata.obs[filters].biosample_accession)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        kf.get_n_splits(biosample_accessions)
        folds_experiment = [list(fold) for _, fold in kf.split(biosample_accessions)]
        folds = []
        for fold in folds_experiment:
            biosample_accessions_fold = biosample_accessions[fold]
            fold = list(np.where([experiment_accession in biosample_accessions_fold for experiment_accession in adata.obs[filters].biosample_accession])[0])
            folds += [fold]

        # Outer cross-validation loop for each fold
        for fold_number, test_fold in enumerate(folds):
            folds_without_test = [item for item in folds if item != test_fold]
            train_fold = [item for sublist in folds_without_test for item in sublist]

            X_train = X_histone[train_fold]
            y_train = y_histone[train_fold]
            X_test = X_histone[test_fold]
            y_test = y_histone[test_fold]

            errors = []
            
            # Inner cross-validation loop for hyperparameter tuning
            for hyperparam in hyperparams:
                y_hat_val = []
                for val_test_fold in folds_without_test:

                    # Select validation train and test folds
                    folds_without_val_test = [n for n in folds_without_test if n != val_test_fold]
                    val_train_fold = [item for sublist in folds_without_val_test for item in sublist]

                    X_val_train = X_histone[val_train_fold]
                    y_val_train = y_histone[val_train_fold]
                    X_val_test = X_histone[val_test_fold]
                    y_val_test = y_histone[val_test_fold]

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
                    adata.obsm['cv_val_df']['cv_val' + str(fold_number+1)][np.where(filters)[0][val_test_fold]] = y_hat_val_test

                # Compute mean squared error and store it
                errors += [np.mean(np.abs(y_train - y_hat_val)**2)]

            # Select the best hyperparameter and train the final model with it
            best_hyperparam = hyperparams[np.where(errors == np.min(errors))[0][0]]
            feature_selector = ElasticNet(alpha=best_hyperparam, max_iter=1000, l1_ratio=0.9, random_state=42)
            feature_selector.fit(X_train, y_train)
            features = (np.abs(feature_selector.coef_) > 0)

            # Dimensionality reduction using TruncatedSVD
            dim_reduction = TruncatedSVD(n_components=np.min([len(X_train)-1, np.sum(features)-1]))
            X_train = dim_reduction.fit_transform(X_train[:,features])
            X_test = dim_reduction.transform(X_test[:,features])

            # Model training using ARDRegression
            model = ARDRegression(n_iter=1000)
            model.fit(X_train, y_train)

            # Model predictions and standard deviation
            y_hat_test, y_hat_std_test = model.predict(X_test, return_std=True)

            # Store paths and models
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
            adata.obsm['cv_test_df']['cv_test' + str(fold_number+1)][np.where(filters)[0][test_fold]] = y_hat_test
            adata.obsm['cv_test_std_df']['cv_test_std' + str(fold_number+1)][np.where(filters)[0][test_fold

        # Clear output to keep the terminal clean
        clear_output()

    return adata

# Read the gene data and filter out the observations related to cancer
adata_genes = sc.read_h5ad('data/encode_data/adata_genes.h5ad')
adata_genes._inplace_subset_obs([not x for x in adata_genes.obs['cancer']])
adata_genes = adata_genes[adata_genes.obs['age'] >= 18]

# Perform nested cross-validation on the gene data and save the results
adata_genes = nested_cross_validation(
    adata=adata_genes, 
    n_folds=10,
    hyperparams=[0.1, 0.05, 0.01, 0.005, 0.001],
)
adata_genes.write_h5ad('results/adata_genes_adult_only.h5ad')
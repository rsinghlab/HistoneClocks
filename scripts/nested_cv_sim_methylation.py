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

def nested_cross_validation_simulation_for_methylation(adata, adata_histone, n_folds=5, hyperparams=[0.05, 0.01], n_sims=2000):
    """
    This function performs nested cross-validation on methylation data with simulations, taking into account the unique histones.
    The cross-validation is performed over different hyperparameters, and the results are stored within the input AnnData object.

    Arguments:
    adata: AnnData object containing the methylation data matrix 'X' and observations 'obs'.
    adata_histone: AnnData object containing histone data.
    n_folds: int, number of folds for cross-validation.
    hyperparams: list, hyperparameters to be tuned.
    n_sims: int, number of simulations.

    Outputs:
    adata: Modified AnnData object including cross-validation results for simulations.
    """
    
    np.random.seed(42)
    seeds = np.random.randint(0, 424242, n_sims)

    # Extracting methylation data and target variable
    X = adata.X
    y = adata.obs['age']

    # Loop through simulations
    for i in tqdm(range(len(seeds))):

        seed = seeds[i]

        # Loop through unique histones
        for histone in np.unique(adata_histone.obs.histone):

            sim_size = adata_histone[adata_histone.obs.histone == histone].shape[0]

            # Initialize data frames to store cross-validation results for each simulation and histone
            adata.obsm['cv_val_df_sim_' + histone + '_' + str(seed)] = pd.DataFrame(columns=['cv_val' + str(i+1) for i in range(n_folds)], index=adata.obs_names, dtype=float)
            adata.obsm['cv_test_df_sim_' + histone + '_' + str(seed)] = pd.DataFrame(columns=['cv_test' + str(i+1) for i in range(n_folds)], index=adata.obs_names, dtype=float)
            adata.obsm['cv_test_std_df_sim_' + histone + '_' + str(seed)] = pd.DataFrame(columns=['cv_test_std' + str(i+1) for i in range(n_folds)], index=adata.obs_names, dtype=float)

            np.random.seed(seed)
            filters = np.array([False]*X.shape[0])
            filters[np.random.randint(0, adata.shape[0], sim_size)] = True

            # Simulated data
            X_sim = X[filters]
            y_sim = y[filters]

            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            kf.get_n_splits(X_sim)
            folds = [list(fold) for _, fold in kf.split(X_sim)]

            # Outer cross-validation loop for each fold
            for fold_number, test_fold in enumerate(folds):

                folds_without_test = [item for item in folds if item != test_fold]
                train_fold = [item for sublist in folds_without_test for item in sublist]

                X_train = X_sim[train_fold]
                y_train = y_sim[train_fold]
                X_test = X_sim[test_fold]
                y_test = y_sim[test_fold]

                errors = []

                # Inner cross-validation loop for hyperparameter tuning
                for hyperparam in hyperparams:

                    y_hat_val = []

                    # Loop through validation folds
                    for val_test_fold in folds_without_test:

                        folds_without_val_test = [n for n in folds_without_test if n != val_test_fold]
                        val_train_fold = [item for sublist in folds_without_val_test for item in sublist]

                        X_val_train = X_sim[val_train_fold]
                        y_val_train = y_sim[val_train_fold]
                        X_val_test = X_sim[val_test_fold]
                        y_val_test = y_sim[val_test_fold]

                        # Feature selection and dimensionality reduction using ElasticNet
                        feature_selector = ElasticNet(alpha=hyperparam, max_iter=1000, l1_ratio=0.9, random_state=42)
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

                        adata.obsm['cv_val_df_sim_' + histone + '_' + str(seed)]['cv_val' + str(fold_number+1)][np.where(filters)[0][val_test_fold]] = y_hat_val_test

                    # Compute mean squared error and store it
                    errors += [np.mean(np.abs(y_train - y_hat_val)**2)]

                # Select best hyperparameter and train the final model with it
                best_hyperparam = hyperparams[np.where(errors == np.min(errors))[0][0]]

                feature_selector = ElasticNet(alpha=best_hyperparam, max_iter=1000, l1_ratio=0.9, random_state=42)
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
                adata.uns['cv_test'+str(fold_number)+'_feature_selector_sim' + str(seed)] = feature_selector_path
                joblib.dump(feature_selector, feature_selector_path)

                dim_reduction_path = 'results/adata_dependencies/' + uuid.uuid4().hex[:20]
                adata.uns['cv_test'+str(fold_number)+'_dim_reduction_sim' + str(seed)] = dim_reduction_path
                joblib.dump(dim_reduction, dim_reduction_path)

                model_path = 'results/adata_dependencies/' + uuid.uuid4().hex[:20]
                adata.uns['cv_test'+str(fold_number)+'_model_sim' + str(seed)] = model_path
                joblib.dump(model, model_path)

                adata.uns['sim_size_sim' + str(seed)] = sim_size
                adata.obsm['cv_test_df_sim_' + histone + '_' + str(seed)]['cv_test' + str(fold_number+1)][np.where(filters)[0][test_fold]] = y_hat_test
                adata.obsm['cv_test_std_df_sim_' + histone + '_' + str(seed)]['cv_test_std' + str(fold_number+1)][np.where(filters)[0][test_fold]] = y_hat_std_test

        # Clear output to keep the terminal clean
        clear_output()

    return adata

# Read the methylation data related to AltumAge
adata_methylation_AltumAge = sc.read_h5ad('data/methylation_data/adata_methylation.h5ad')

# Read the gene data
adata_genes = sc.read_h5ad('data/encode_data/adata_genes.h5ad')

# Perform nested cross-validation simulation for methylation, using AltumAge methylation data and gene data,
# with specified number of folds, hyperparameters, and simulations
adata_methylation_AltumAge_simulations = nested_cross_validation_simulation_for_methylation(
    adata=adata_methylation_AltumAge,
    adata_histone=adata_genes,
    n_folds=5,
    hyperparams=[0.1, 0.05, 0.01, 0.005, 0.001],
    n_sims=100,
)

# Save the simulation results to an H5AD file
adata_methylation_AltumAge_simulations.write_h5ad('results/adata_methylation_AltumAge_simulations.h5ad')

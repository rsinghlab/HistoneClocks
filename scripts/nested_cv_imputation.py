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

def nested_cross_validation_with_imputed_data(adata, adata_imputation, n_folds=5, hyperparams=[0.05, 0.01]):
    """
    Perform nested cross-validation for a given dataset with the addition of imputed data.
    
    Arguments:
    adata : AnnData - The main data object.
    adata_imputation : AnnData - The data object containing imputation information.
    n_folds : int - The number of folds to use in cross-validation (default is 5).
    hyperparams : list - List of hyperparameters to tune for the model (default is [0.05, 0.01]).
    
    Outputs:
    adata : AnnData - The updated data object with cross-validation results stored in obsm.
    """
    
    # Create dataframes to store cross-validation results
    adata.obsm['cv_val_df'] = pd.DataFrame(columns=['cv_val' + str(i+1) for i in range(n_folds)], index=adata.obs_names, dtype=float)
    adata.obsm['cv_test_df'] = pd.DataFrame(columns=['cv_test' + str(i+1) for i in range(n_folds)], index=adata.obs_names, dtype=float)
    adata.obsm['cv_test_std_df'] = pd.DataFrame(columns=['cv_test_std' + str(i+1) for i in range(n_folds)], index=adata.obs_names, dtype=float)
    
    # Extract main features
    X = adata.X
    y = adata.obs['age']
    
    # Extract imputation features
    X_imputation = adata_imputation.X
    y_imputation = adata_imputation.obs['age']
    
    # Iterate over each unique histone in the data
    for i in tqdm(range(len(np.unique(adata.obs['histone'])))):
        
        histone = np.unique(adata.obs['histone'])[i]

        # Apply filters to data based on histone
        filters = (adata.obs['histone'] == histone)
        X_histone = X[filters]
        y_histone = y[filters]  
        X_histone_imputation = X_imputation[adata_imputation.obs['histone'] == histone]
        y_histone_imputation = y_imputation[adata_imputation.obs['histone'] == histone]  
        
        # Divide folds by biosample, making sure replicates are together to not inflate score
        biosample_accessions = np.unique(adata.obs[filters].biosample_accession)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        kf.get_n_splits(biosample_accessions)
        folds_experiment = [list(fold) for _, fold in kf.split(biosample_accessions)]
        folds = []
        for fold in folds_experiment:
            biosample_accessions_fold = biosample_accessions[fold]
            fold = list(np.where([experiment_accession in biosample_accessions_fold for experiment_accession in adata.obs[filters].biosample_accession])[0])
            folds += [fold]
        
        # Iterate over each fold for cross-validation
        for fold_number, test_fold in enumerate(folds):
            
            # Identify training and test folds
            folds_without_test = [item for item in folds if item != test_fold]
            train_fold = [item for sublist in folds_without_test for item in sublist]
            X_train = X_histone[train_fold]
            y_train = y_histone[train_fold]
            X_test = X_histone[test_fold]
            y_test = y_histone[test_fold]
            
            # Perform hyperparameter tuning using validation set
            errors = []
            for hyperparam in hyperparams:
                
                y_hat_val = []
                for val_test_fold in folds_without_test:

                    # Identify validation training and test sets
                    folds_without_val_test = [n for n in folds_without_test if n != val_test_fold]
                    val_train_fold = [item for sublist in folds_without_val_test for item in sublist]
                    X_val_train = X_histone[val_train_fold]
                    y_val_train = y_histone[val_train_fold]
                    
                    # Concatenate training data with imputation
                    X_val_train = np.concatenate([X_val_train, X_histone_imputation], axis=0)
                    y_val_train = np.concatenate([y_val_train, y_histone_imputation], axis=0)
                    X_val_test = X_histone[val_test_fold]
                    y_val_test = y_histone[val_test_fold]
                                        
                    # Feature selection and dimensionality reduction
                    feature_selector =  ElasticNet(alpha=hyperparam, max_iter=1000, l1_ratio=0.9, random_state=42)
                    feature_selector.fit(X_val_train, y_val_train)
                    features = (np.abs(feature_selector.coef_) > 0)
                    dim_reduction = TruncatedSVD(n_components=np.min([len(X_val_train)-1, np.sum(features)-1]))
                    X_val_train = dim_reduction.fit_transform(X_val_train[:,features])
                    X_val_test = dim_reduction.transform(X_val_test[:,features])

                    # Model training and prediction
                    model = ARDRegression(n_iter=1000)
                    model.fit(X_val_train, y_val_train)
                    y_hat_val_test = model.predict(X_val_test)
                    y_hat_val += list(y_hat_val_test)

                    # Store validation results
                    adata.obsm['cv_val_df']['cv_val' + str(fold_number+1)][np.where(filters)[0][val_test_fold]] = y_hat_val_test
                    
                # Compute and store errors for hyperparameter tuning
                errors += [np.mean(np.abs(y_train - y_hat_val)**2)]
                
            # Select the best hyperparameter based on validation error
            best_hyperparam = hyperparams[np.where(errors == np.min(errors))[0][0]]
            
            # Concatenate training data with imputation
            X_train = np.concatenate([X_train, X_histone_imputation], axis=0)
            y_train = np.concatenate([y_train, y_histone_imputation], axis=0)
            
            # Feature selection and dimensionality reduction with best hyperparameter
            feature_selector =  ElasticNet(alpha=best_hyperparam, max_iter=1000, l1_ratio=0.9, random_state=42)
            feature_selector.fit(X_train, y_train)
            features = (np.abs(feature_selector.coef_) > 0)
            dim_reduction = TruncatedSVD(n_components=np.min([len(X_train)-1, np.sum(features)-1]))
            X_train = dim_reduction.fit_transform(X_train[:,features])
            X_test = dim_reduction.transform(X_test[:,features])

            # Final model training and prediction
            model = ARDRegression(n_iter=1000)
            model.fit(X_train, y_train)
            y_hat_test, y_hat_std_test = model.predict(X_test, return_std=True)
            
            # Save the model components
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
            adata.obsm['cv_test_df']['cv_test' + str(fold_number+1)][np.where(filters)[0][test_fold]] = y_hat_test
            adata.obsm['cv_test_std_df']['cv_test_std' + str(fold_number+1)][np.where(filters)[0][test_fold]] = y_hat_std_test
            
        # Clear output for next iteration
        clear_output()
            
    return adata

# Read the adata_genes file and subset observations by excluding those marked as cancer
adata_genes = sc.read_h5ad('data/encode_data/adata_genes.h5ad')
adata_genes._inplace_subset_obs([not x for x in adata_genes.obs['cancer']])

# Read the adata_gaps file and subset observations by excluding those marked as cancer
adata_gaps = sc.read_h5ad('data/encode_data/adata_gaps.h5ad')
adata_gaps._inplace_subset_obs([not x for x in adata_gaps.obs['cancer']])

# Concatenate adata_genes and adata_gaps and subset observations again to exclude cancer
adata_whole_genome = anndata.concat([adata_genes, adata_gaps], axis=1, join='outer')
adata_whole_genome.obs = adata_genes.obs.copy()
adata_whole_genome._inplace_subset_obs([not x for x in adata_whole_genome.obs['cancer']])

# Read the adata_hm27_450_cpgs file and subset observations by excluding those marked as cancer
adata_hm27_450_cpgs = sc.read_h5ad('data/encode_data/adata_hm27_450_cpgs.h5ad')
adata_hm27_450_cpgs._inplace_subset_obs([not x for x in adata_hm27_450_cpgs.obs['cancer']])

# Read horvath_cpgs data, subset adata_hm27_450_cpgs using it, and exclude cancer observations
horvath_cpgs = pd.read_pickle('data/methylation_data/horvath_cpgs.pkl').tolist()
adata_horvath_cpgs = adata_hm27_450_cpgs[:,horvath_cpgs].copy()
adata_horvath_cpgs._inplace_subset_obs([not x for x in adata_horvath_cpgs.obs['cancer']])

# Read the adata_genes_imputation file and subset observations by excluding those marked as cancer
adata_genes_imputation = sc.read_h5ad('data/encode_data/adata_genes_imputation.h5ad')
adata_genes_imputation._inplace_subset_obs([not x for x in adata_genes_imputation.obs['cancer']])

# Read the adata_gaps_imputation file and subset observations by excluding those marked as cancer
adata_gaps_imputation = sc.read_h5ad('data/encode_data/adata_gaps_imputation.h5ad')
adata_gaps_imputation._inplace_subset_obs([not x for x in adata_gaps_imputation.obs['cancer']])

# Concatenate adata_genes_imputation and adata_gaps_imputation and subset observations again to exclude cancer
adata_whole_genome_imputation = anndata.concat([adata_genes_imputation, adata_gaps_imputation], axis=1, join='outer')
adata_whole_genome_imputation.obs = adata_genes_imputation.obs.copy()
adata_whole_genome_imputation._inplace_subset_obs([not x for x in adata_whole_genome_imputation.obs['cancer']])

# Read the adata_hm27_450_cpgs_imputation file and subset observations by excluding those marked as cancer
adata_hm27_450_cpgs_imputation = sc.read_h5ad('data/encode_data/adata_hm27_450_cpgs_imputation.h5ad')
adata_hm27_450_cpgs_imputation._inplace_subset_obs([not x for x in adata_hm27_450_cpgs_imputation.obs['cancer']])

# Read horvath_cpgs data again, subset adata_hm27_450_cpgs_imputation using it, and exclude cancer observations
horvath_cpgs = pd.read_pickle('data/methylation_data/horvath_cpgs.pkl').tolist()
adata_horvath_cpgs_imputation = adata_hm27_450_cpgs_imputation[:,horvath_cpgs].copy()
adata_horvath_cpgs_imputation._inplace_subset_obs([not x for x in adata_horvath_cpgs.obs['cancer']])

# Perform nested cross-validation for various datasets and hyperparameters and write the results

# Nested cross-validation on adata_genes
adata_genes = nested_cross_validation_with_imputed_data(
    adata=adata_genes, 
    adata_imputation=adata_genes_imputation,
    n_folds=10,
    hyperparams=[0.1, 0.05, 0.01, 0.005, 0.001],
)
adata_genes.write_h5ad('results/adata_genes_imputation.h5ad')

# Nested cross-validation on adata_gaps
adata_gaps = nested_cross_validation_with_imputed_data(
    adata=adata_gaps, 
    adata_imputation=adata_gaps_imputation,
    n_folds=10,
    hyperparams=[0.1, 0.05, 0.01, 0.005, 0.001],
)
adata_gaps.write_h5ad('results/adata_gaps_imputation.h5ad')

# Nested cross-validation on adata_whole_genome
adata_whole_genome = nested_cross_validation_with_imputed_data(
    adata=adata_whole_genome, 
    adata_imputation=adata_whole_genome_imputation,
    n_folds=10,
    hyperparams=[0.1, 0.05, 0.01, 0.005, 0.001],
)
adata_whole_genome.write_h5ad('results/adata_whole_genome_imputation.h5ad')

# Nested cross-validation on adata_hm27_450_cpgs
adata_hm27_450_cpgs = nested_cross_validation_with_imputed_data(
    adata=adata_hm27_450_cpgs, 
    adata_imputation=adata_hm27_450_cpgs_imputation,
    n_folds=10,
    hyperparams=[0.1, 0.05, 0.01, 0.005, 0.001],
)
adata_hm27_450_cpgs.write_h5ad('results/adata_hm27_450_cpgs_imputation.h5ad')

# Nested cross-validation on adata_horvath_cpgs
adata_horvath_cpgs = nested_cross_validation_with_imputed_data(
    adata=adata_horvath_cpgs, 
    adata_imputation=adata_horvath_cpgs_imputation, 
    n_folds=10,
    hyperparams=[0.1, 0.05, 0.01, 0.005, 0.001],
)
adata_horvath_cpgs.write_h5ad('results/adata_horvath_cpgs_imputation.h5ad')
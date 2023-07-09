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

def train_histone_models_for_analysis(adata, n_folds=5, hyperparams=[0.05, 0.01]):
            
    X = adata.X
    y = adata.obs['age']
    
    for i in tqdm(range(len(np.unique(adata.obs['histone'])))):
        
        histone = np.unique(adata.obs['histone'])[i]

        filters = (adata.obs['histone'] == histone)
                
        X_histone = X[filters]
        y_histone = y[filters]   
                
        #divide folds by experiment, making sure replicates are together to not inflate score
        experiment_accessions = np.unique(adata.obs[filters].experiment_accession)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        kf.get_n_splits(experiment_accessions)
        folds_experiment = [list(fold) for _, fold in kf.split(experiment_accessions)]
        folds = []
        for fold in folds_experiment:
            experiment_accessions_fold = experiment_accessions[fold]
            fold = list(np.where([experiment_accession in experiment_accessions_fold for experiment_accession in adata.obs[filters].experiment_accession])[0])
            folds += [fold]
                    
        errors = []
        for hyperparam in hyperparams:

            y_hat = []
            for fold_number, test_fold in zip(range(n_folds), folds):

                folds_without_test = [item for item in folds if item != test_fold]
                train_fold = [item for sublist in folds_without_test for item in sublist]

                X_train = X_histone[train_fold]
                y_train = y_histone[train_fold]

                X_test = X_histone[test_fold]
                y_test = y_histone[test_fold]

                feature_selector =  ElasticNet(alpha=hyperparam, max_iter=2000, l1_ratio=0.9, random_state=42)
                feature_selector.fit(X_train, y_train)

                features = (np.abs(feature_selector.coef_) > 0)

                dim_reduction = TruncatedSVD(n_components=np.min([len(X_train)-1, np.sum(features)-1]))

                X_train = dim_reduction.fit_transform(X_train[:,features])
                X_test = dim_reduction.transform(X_test[:,features])

                model = ARDRegression(n_iter=2000)
                model.fit(X_train, y_train)

                y_hat_test = model.predict(X_test)
                y_hat += list(y_hat_test)

            errors += [np.mean(np.abs(y_histone - y_hat)**2)]
                
        best_hyperparam = hyperparams[np.where(errors == np.min(errors))[0][0]]

        feature_selector =  ElasticNet(alpha=best_hyperparam, max_iter=2000, l1_ratio=0.9, random_state=42)
        feature_selector.fit(X_histone, y_histone)

        features = (np.abs(feature_selector.coef_) > 0)

        dim_reduction = TruncatedSVD(n_components=np.min([len(X_histone)-1, np.sum(features)-1]))
        X_histone = dim_reduction.fit_transform(X_histone[:,features])

        model = ARDRegression(n_iter=2000)
        model.fit(X_histone, y_histone)

        feature_selector_path = 'results/models/' + histone + '_feature_selector.pkl'
        joblib.dump(feature_selector, feature_selector_path)

        dim_reduction_path = 'results/models/' + histone + '_dim_reduction.pkl'
        joblib.dump(dim_reduction, dim_reduction_path)

        model_path = 'results/models/' + histone + '_model.pkl'
        joblib.dump(model, model_path)
            
        clear_output()
        
adata_genes = sc.read_h5ad('data/encode_data/adata_genes.h5ad')
adata_genes._inplace_subset_obs([not x for x in adata_genes.obs['cancer']])
        
train_histone_models_for_analysis(
    adata=adata_genes, 
    n_folds=10,
    hyperparams=[0.1, 0.05, 0.01, 0.005, 0.001],
)
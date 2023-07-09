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

def nested_cross_validation_with_noise(adata, n_folds=5, hyperparams=[0.05, 0.01], gaussian_noises=[0, 0.5, 1]):
    
    X = adata.X
    y = adata.obs['age']
    
    for noise in gaussian_noises:
        adata.obsm[str(noise) + 'cv_test_df'] = pd.DataFrame(columns=['cv_test' + str(i+1) for i in range(n_folds)], index=adata.obs_names, dtype=float)
        adata.obsm[str(noise) + 'cv_test_std_df'] = pd.DataFrame(columns=['cv_test_std' + str(i+1) for i in range(n_folds)], index=adata.obs_names, dtype=float)

    for i in tqdm(range(len(np.unique(adata.obs['histone'])))):

        histone = np.unique(adata.obs['histone'])[i]

        filters = (adata.obs['histone'] == histone)

        X_histone = X[filters]
        y_histone = y[filters]   

        #divide folds by experiment, making sure replicates are together to not inflate score
        experiment_accessions = np.unique(adata.obs[filters].biosample_accession)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        kf.get_n_splits(experiment_accessions)
        folds_experiment = [list(fold) for _, fold in kf.split(experiment_accessions)]
        folds = []
        for fold in folds_experiment:
            experiment_accessions_fold = experiment_accessions[fold]
            fold = list(np.where([experiment_accession in experiment_accessions_fold for experiment_accession in adata.obs[filters].biosample_accession])[0])
            folds += [fold]

        for fold_number, test_fold in enumerate(folds):

            folds_without_test = [item for item in folds if item != test_fold]
            train_fold = [item for sublist in folds_without_test for item in sublist]

            X_train = X_histone[train_fold] 
            y_train = y_histone[train_fold]

            X_test = X_histone[test_fold]
            y_test = y_histone[test_fold]

            errors = []
            for hyperparam in hyperparams:

                y_hat_val = []
                for val_test_fold in folds_without_test:

                    folds_without_val_test = [n for n in folds_without_test if n != val_test_fold]
                    val_train_fold = [item for sublist in folds_without_val_test for item in sublist]

                    X_val_train = X_histone[val_train_fold]
                    y_val_train = y_histone[val_train_fold]

                    X_val_test = X_histone[val_test_fold]
                    y_val_test = y_histone[val_test_fold]

                    feature_selector =  ElasticNet(alpha=hyperparam, max_iter=1000, l1_ratio=0.9, random_state=42)
                    feature_selector.fit(X_val_train, y_val_train)

                    features = (np.abs(feature_selector.coef_) > 0)

                    dim_reduction = TruncatedSVD(n_components=np.min([len(X_val_train)-1, np.sum(features)-1]))

                    X_val_train = dim_reduction.fit_transform(X_val_train[:,features])
                    X_val_test = dim_reduction.transform(X_val_test[:,features])

                    model = ARDRegression(n_iter=1000)
                    model.fit(X_val_train, y_val_train)

                    y_hat_val_test = model.predict(X_val_test)
                    y_hat_val += list(y_hat_val_test)

                errors += [np.mean(np.abs(y_train - y_hat_val)**2)]

            best_hyperparam = hyperparams[np.where(errors == np.min(errors))[0][0]]

            feature_selector =  ElasticNet(alpha=best_hyperparam, max_iter=1000, l1_ratio=0.9, random_state=42)
            feature_selector.fit(X_train, y_train)

            features = (np.abs(feature_selector.coef_) > 0)

            dim_reduction = TruncatedSVD(n_components=np.min([len(X_train)-1, np.sum(features)-1]))
            X_train = dim_reduction.fit_transform(X_train[:,features])

            model = ARDRegression(n_iter=1000)
            model.fit(X_train, y_train)
            
            for noise in gaussian_noises:
                
                X_test_noise = X_test + noise*np.random.normal(loc=0.0, scale=np.std(X_test.flatten()), size=X_test.shape)
                X_test_noise = dim_reduction.transform(X_test_noise[:,features])
                y_hat_test, y_hat_std_test = model.predict(X_test_noise, return_std=True)

                feature_selector_path = 'results/adata_dependencies/' + uuid.uuid4().hex[:20]
                adata.uns[str(noise) + 'cv_test'+str(fold_number)+'_feature_selector'] = feature_selector_path
                joblib.dump(feature_selector, feature_selector_path)

                dim_reduction_path = 'results/adata_dependencies/' + uuid.uuid4().hex[:20]
                adata.uns[str(noise) + 'cv_test'+str(fold_number)+'_dim_reduction'] = dim_reduction_path
                joblib.dump(dim_reduction, dim_reduction_path)

                model_path = 'results/adata_dependencies/' + uuid.uuid4().hex[:20]
                adata.uns[str(noise) + 'cv_test'+str(fold_number)+'_model'] = model_path
                joblib.dump(model, model_path)

                adata.obsm[str(noise) + 'cv_test_df']['cv_test' + str(fold_number+1)][np.where(filters)[0][test_fold]] = y_hat_test
                adata.obsm[str(noise) + 'cv_test_std_df']['cv_test_std' + str(fold_number+1)][np.where(filters)[0][test_fold]] = y_hat_std_test

        clear_output()
            
    return adata

adata_genes = sc.read_h5ad('data/encode_data/adata_genes.h5ad')
adata_genes._inplace_subset_obs([not x for x in adata_genes.obs['cancer']])

adata_genes_noise = nested_cross_validation_with_noise(
    adata=adata_genes, 
    n_folds=10,
    hyperparams=[0.1, 0.05, 0.01, 0.005, 0.001],
    gaussian_noises=[0, 0.3, 0.6, 0.9, 1.2, 1.5],
)
adata_genes.write_h5ad('results/adata_genes_gaussian_noise.h5ad')

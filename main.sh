#!/bin/bash

#change to home directory
cd SageMaker

#create necessary folders
mkdir metadata
mkdir metadata/methylation_data #no cloud link for this
mkdir data/encode_data/bigWigs -p
mkdir results
mkdir results/adata_dependencies
mkdir results/models
mkdir results/enrichment
mkdir figures

#download Ensembl 105 from an R script
source activate R
Rscript scripts/Ensembl-105-EnsDb-for-Homo-sapiens.r
conda deactivate

#download metadata for the 27k and 450k Illumina DNA methylation arrays
wget https://zhouserver.research.chop.edu/InfiniumAnnotation/20180909/HM27/HM27.hg38.manifest.tsv.gz -P metadata
wget https://zhouserver.research.chop.edu/InfiniumAnnotation/20180909/HM450/HM450.hg38.manifest.tsv.gz -P metadata

#install required python packages
source activate python3
pip install -r requirements.txt
conda install -c conda-forge -c bioconda gseapy -y
conda install -c bioconda pybigwig -y

#download data
python scripts/download_and_process_methylation.py
python scripts/download_and_process_geo.py
python scripts/download_and_process_encode.py
python scripts/download_and_process_encode_imputation.py

#run scripts
python scripts/nested_cv.py
python scripts/nested_cv_sim_methylation.py
python scripts/nested_cv_pan_histone.py
python scripts/nested_cv_noise.py
python scripts/nested_cv_across_histones.py
python scripts/nested_cv_across_histones_flipped.py
python scripts/nested_cv_across_histone_group.py
python scripts/train_histone_models.py
python scripts/train_pan_histone_model.py

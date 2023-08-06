[![Beta](https://img.shields.io/badge/status-beta-yellow)](https://github.com/rsinghlab/HistoneClocks)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![GitHub stars](https://img.shields.io/github/stars/rsinghlab/HistoneClocks)](https://github.com/rsinghlab/HistoneClocks/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/rsinghlab/HistoneClocks)](https://github.com/rsinghlab/HistoneClocks/network/members)
![Downloads](https://img.shields.io/github/downloads/rsinghlab/HistoneClocks/total)

# 🧬🕡 Histone mark age of human tissues and cells 🕡🧬

For the link to our preprint, please click [here](URLXXXXXXX).


## 🚀 Usage

To replicate the results, please spin up an AWS SageMaker instance (in our case, we used ml.t3.2xlarge), clone this repository, and run main.sh. If you already have already downloaded the processed ENCODE data (see below), then feel free to comment out the scripts used for downloads.

---

### 🏛️ Project Structure

Here is a brief description of the main folders and files in this repository:

- **`analysis/`**: Folder containing the analysis notebook files for the four main sections of the manuscript.
- **`metadata/`**: Folder containing the required metadata to run the scripts.
- **`results/models/`**: Folder containing the feature selection, dimensionality reduction, and ARD regressor for each histone mark age predictor and for the pan histone age predictor.
- **`scripts/`**: Folder containing all of the scripts required to reproduce the results of the paper.
- **`tutorial/`**: Folder containing the tutorial notebook with the functions to predict the histone mark age of your own data.
- **`.gitignore`**: File to specify files that Git should ignore (e.g. .DS_Store files).
- **`LICENSE`**: License file.
- **`README.md`**: This README file.
- **`main.sh`**: Main shell script that calls other scripts in the scripts folder.
- **`requirements.txt`**: File containing a list of items to be installed through main.sh.


### 💡 Tutorial on Using the Histone Mark Age Predictors

The tutorial below provides a step-by-step guide to using the histone mark age predictors. A more detailed version is also available in the Jupyter notebook inside the 'tutorial' folder named 'tutorial.ipynb'.

1. **Load required packages:**
   Import the necessary packages for data processing and prediction.

2. **Download an Example File from the ENCODE Project:**
   Execute the command to download a training sample (bigWig file) used in the models. Alternatively, if you would like to use your own ChIP-Seq data, please refer to the [ENCODE website](https://www.encodeproject.org) for guidelines to handle your biosample appropriately and to the [ENCODE ChIP-Seq pipeline GitHub](https://github.com/ENCODE-DCC/chip-seq-pipeline2) to obtain the bigWig file from the sequencing data.

3. **Process the bigWig File:**
   Use the function \`process_bigWig(bigWig_file_path, annotation_file_path)\` to extract genomic annotations and transform signal values.

4. **Predict the Histone Mark Age:**
   Utilize the function \`predict_histone_mark_age(processed_sample, histone)\` to predict age based on the processed sample for a given histone type.

5. **Print the Result:**
   The code will print the predicted histone mark age.

Example code snippet:
```python
sample = process_bigWig('ENCFF386QWG.bigWig')
histone_mark = 'H3K4me3'
y_hat = predict_histone_mark_age(sample, histone=histone_mark)[0]
print(f'The predicted {histone_mark} age is {round(y_hat,3)} years.')
```

## 📦 Data availability

All data used was publicly available from the ENCODE project. This can be programmatically accessed and downloaded through the scripts in this GitHub. Nevertheless, to download the already-processed data with the results, please access our Zenodo repository [here](URLXXXXXXX). This should make it easier to train any future models. 

## 📝 Citation

To cite our study, please use the following:

de Lima Camillo, L.P., Lapierre, L.R. & Singh, R. Histone mark age of human tissues and cells. bioRxiv X, X (2023). [URLXXXXX](URLXXXXX)

BibTex citation:
```
@article {de_Lima_Camillo_HistoneClocks,
	author = {de Lima Camillo, Lucas Paulo and Lapierre, Louis R and Singh, Ritambhara},
	title = {Histone mark age of human tissues and cells},
	year = {2023},
	doi = {XXXXX},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {XXXXX},
	eprint = {XXXXX},
	journal = {bioRxiv}
}
```

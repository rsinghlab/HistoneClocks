[![Beta](https://img.shields.io/badge/status-beta-yellow)](https://github.com/rsinghlab/HistoneClocks)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![GitHub stars](https://img.shields.io/github/stars/rsinghlab/HistoneClocks)](https://github.com/rsinghlab/HistoneClocks/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/rsinghlab/HistoneClocks)](https://github.com/rsinghlab/HistoneClocks/network/members)
![Downloads](https://img.shields.io/github/downloads/rsinghlab/HistoneClocks/total)

# 🧬🕡 Histone Mark Age of Human Tissues and Cells 🕡🧬

This repository hosts the code and resources for predicting histone mark age in human tissues and cells. Our preprint detailing the methodology and results can be found [here](URLXXXXXXX).

## 🏛️ Project Structure

- **`analysis/`**: Analysis notebook files for the manuscript's four main sections [for peer-review].
- **`metadata/`**: Metadata required to run the scripts.
- **`results/models/`**: Feature selection, dimensionality reduction, and ARD regressor for each histone mark age predictor and the pan histone age predictor.
- **`scripts/`**: Scripts to reproduce the paper's results.
- **`tutorial/`**: Tutorial notebook for predicting histone mark age with your own data.
- **`main.sh`**: Main shell script calling other scripts in the scripts folder.
- **`requirements.txt`**: Required dependencies.
- **`LICENSE`**: License file.
- **`.gitignore`**: Files to ignore in Git (e.g., .DS_Store).
- **`README.md`**: This README file.

## 🚀 Usage

1. **Set up Environment**: Spin up an AWS SageMaker instance (e.g., `ml.t3.2xlarge`) or any other computer.
2. **Clone Repository**: Clone this repository to your environment.
3. **Download Processed ENCODE Data** (optional): Access our [Google Drive](https://drive.google.com/drive/u/2/folders/1mhpAH_bXOyutNfWi0VZdQi0DvQOa7edN). Copy the `data` folder and all files within to the root of your directory. If you've already downloaded the processed ENCODE data, comment out the download scripts.
4. **Run `main.sh`**: Run the `main.sh` script to replicate the results.

## 💡 Tutorial on Using the Histone Mark Age Predictors

Follow this step-by-step guide to use the histone mark age predictors. A more detailed version is available in the 'tutorial' folder named 'tutorial.ipynb'.

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

All data used was publicly available from the ENCODE project. This can be programmatically accessed and downloaded through the scripts in this GitHub. Nevertheless, to download the already-processed data with the results, please access our [Google Drive](https://drive.google.com/drive/u/2/folders/1mhpAH_bXOyutNfWi0VZdQi0DvQOa7edN). This should make it easier to train any future models. 

## 📝 Citation

To cite our study, please use the following:

de Lima Camillo, L.P., Horvath, S. & Singh, R. Histone mark age of human tissues and cells. bioRxiv X, X (2023). [URLXXXXX](URLXXXXX)

BibTex citation:
```
@article {de_Lima_Camillo_HistoneClocks,
	author = {de Lima Camillo, Lucas Paulo and Horvath, Steve and Singh, Ritambhara},
	title = {Histone mark age of human tissues and cells},
	year = {2023},
	doi = {XXXXX},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {XXXXX},
	eprint = {XXXXX},
	journal = {bioRxiv}
}
```

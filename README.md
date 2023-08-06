# 🧬🕡 Histone mark age of human tissues and cells 🕡🧬

[![Preprint](https://img.shields.io/badge/Preprint-Link-blue)](URL XXXXXXX)

## 🚀 Usage

To replicate the results, please spin up an AWS SageMaker instance (in our case, we used ml.t3.2xlarge), clone this repository, and run main.sh.

---

### 💡 Tutorial on Using the Histone Mark Age Predictors

Follow the guide below or refer to 'tutorial.ipynb' inside the 'tutorial' folder.

1. **Load required packages:** ...
2. **Download an Example File:** ...
3. **Process the bigWig File:** ...
4. **Predict the Histone Mark Age:** ...
5. **Print the Result:** ...

Example code snippet:
```python
sample = process_bigWig('ENCFF386QWG.bigWig')
histone_mark = 'H3K4me3'
y_hat = predict_histone_mark_age(sample, histone=histone_mark)[0]
print(f'The predicted {histone_mark} age is {round(y_hat,3)} years.')
```

## 📦 Data availability

All data used was publicly available from the ENCODE project. This can be programmatically accessed and downloaded through the scripts in this GitHub. Nevertheless, to download the already-processed data with the results, please access our Zenodo repository [here](URLXXXXXXX).

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

# Analysis of the genotype-phenotype map of the PLT-MADS regulatory network

This repository contains the code and data necessary to reproduce the analses and figures presented in the following publication in which we study the genetic architecture of inflorescence branching in tomato by combining genetic perturbations of two pairs of paralog genes in a gene regulatory network. 

- Sophia G. Zebell, Carlos Martí-Gómez, Blaine Fitzgerald, Camila Pinto Da Cunha, Michael Lach, Brooke M. Seman, Anat Hendelman, Simon Sretnovic, Madelaine Bartlett, Yiping Qi, Yuval Eshed, David M. McCandlish*, Zachary B. Lippman (2025). Cryptic variation in a plant regulatory network fuels phenotypic change through hierarchical epistasis

### Requirements

Create a new python environment

```bash
conda create -n tomato python==3.8
conda activate tomato
```

Running the scripts for the analysis requires a series of python libraries 

- numpy
- pandas
- scipy
- statsmodels
- torch
- matplotlib
- seaborn

They can be installed manually using `pip`. To install the exact versions of the library we used:

```bash
pip install -r requirements.txt
```

### Folders

- data: contains the raw and processed data files generated from the analysis
- results: contains result tables from fitting the models and making predictions
- scripts: 
  - models: scripts to preprocess data and fit the different models, evaluate them and make phenotypic predictions under them 
  - figures/main: scripts to generate the panels from the main Figure 4
  - figures/supp: scripts to generate associated supplementary Figures
  - figures/v0: outdated scripts generated during the project for data analysis
- figures: contains the generated main and supplementary figure panels

### Execution

For reproducing the complete analysis, the bash script  `run.sh` runs all the scripts in the specified order.

```bash
bash run.sh
```

However, each script can be run independently by leveraging the preprocessed files that we provide in the `data` and `results` folders




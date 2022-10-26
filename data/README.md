# Data

## Dataset for demonstration

### For documentation.ipynb

id.csv identifies the clone id of the cells, allowing comparison between the original dataset and the predicted dataset.

AD.mtx and DP.mtx are smaller dataset that demonstrates how the Betabin package works.

cellSNP.tag.AD.mtx and cellSNP.tag.DP.mtx are the full dataset that visualizes the accuracy of the Betabin package.

passed_variant_names.txt stores the names of the variant names that can be used to label the bar graph shown in docs/documentation.ipynb.

### For BetaBinomial_example.ipynb

isoform_count_toaod.csv.gz is the dataset for doing betabinomial regression using BetaBinomial package. It gives the data reagarding the disease type and the count in each isoform.  

gene_name_random.csv marks the genes that have been selected to do the regression.

BetaBinomial_result.csv includes the result of parameters, negative log-likelihood given by the BetaBinomial package to the selected genes.

comparison.csv is the comparison of the result between the BetaBinomial package and aod R package (https://cran.r-project.org/web/packages/aod/aod.pdf). 

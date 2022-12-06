# **CARL RNA Environment**

This is the CARL RNA environment that has been adapted from the [Learning to Design RNA](https://openreview.net/pdf?id=ByfyHh05tQ)  by Runge et. al. The code has been adapted from [https://github.com/automl/learna](https://github.com/automl/learna) with a carl wrapper written around the envionment. 

## Datasets
To download and build the datasets we report on in our publications, namely the Rfam-Taneda [[Taneda, 2011]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3169953/pdf/aabc-4-001.pdf) dataset and our three proposed datasets, Rfam-Learn-Train, Rfam-Learn-Validation and Rfam-Learn-Test, run the following command after installation of all requirements.

```
cd data
./download_and_build_rfam_learn.sh
./download_and_build_rfam_taneda.sh
```
This will download all files and save them into the `data/` directory. 
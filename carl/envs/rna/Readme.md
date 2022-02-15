# Downloading RNA data
To use the RNA env, you need to install the base environment as well as download
the sequence data. If you have not downloaded the learna submodule automatically,
please do so now and make sure it is placed in this directory.

There are three different datasets you can download.
First, you need to enter the learna directory:
```
cd learna
```
Then to download the Eterna100 dataset, run:
```
python -m src.data.download_and_build_eterna ./src/data/secondaries_to_single_files.sh data/eterna data/eterna/interim/eterna.txt
```
A note on python requests: the content length header element that's used in this original download
script is not guaranteed to be present. If there's an error, try replacing line 6 in the python scripts with:
```python
response = requests.get(url, stream=True, headers={'Accept-Encoding': None})
```
Afterwards the download should start as normal.

For Rfam-Taneda, use:
```
sh  src/data/download_and_build_rfam_taneda.sh
```
And finally for Rfam-Learn:
```
sh src/data/download_and_build_rfam_learn.sh
mv data/rfam_learn/test data/rfam_learn_test
mv data/rfam_learn/validation data/rfam_learn_validation
mv data/rfam_learn/train data/rfam_learn_train
rm -rf data/rfam_learn
```

If everything went well, you should see .rna files in the target data folders. By default, the env will try to parse them out of carl/envs/rna/learna/data/<datasetname>, so make sure to specify a different path upon initialization if you want to relocate the data. 

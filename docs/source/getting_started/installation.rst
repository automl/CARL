Installing CARL
===============

Base Installation
-----------------
In order to install CARL, first clone the GitHub repository:
::
    git clone https://github.com/automl/CARL.git
    cd CARL

We recommend using a virtual environment for installation, e.g. conda:
::
    conda create -n carl python=3.9
    conda activate carl

The base version of CARL only includes the classic control environments 
and can be installed by running:
::
    pip install .

Additional Environments
-----------------------
The other environments like Bos2D and Brax are optional dependencies, so you can choose which you want
to install. For the full set:
::
    pip install -e .[box2d, brax, rna, mario]

To use ToadGAN, additionally run:
::
    javac src/envs/mario/Mario-AI-Framework/**/*.java

If you plan on using the RNA environment, you need to download the RNA sequence data:
::
    cd src/envs/rna/learna
    make requirements
    make data
   

CARL on Windows & MAC
---------------------
These installation instructions might not work fully on Windows systems and have not
been tested there. For MAC, you will need to install Box2D via conda:
::
    conda install -c conda-forge gym-box2d

We generally test and develop CARL on Linux systems, but aim to keep it as compatible
with MAC as possible. The ToadGAN environment is Linux exclusive at this
point. 
# in order to run this, on the LUIS cluster in the home directory navigate to src and do not have any conda environment activated

# project info
CONDA_ENV_NAME={conda_env_name}
PROJECT_NAME={project_name}
REPO_DIR=$HOME/repos
BRANCH_NAME={branch_name}

# setup environment
cd $HOME
module load Miniconda2/4.7.10
conda activate $CONDA_ENV_NAME

# pull newest changes
cd $REPO_DIR/$PROJECT_NAME
git checkout $BRANCH_NAME
git pull

# create working directories
WORKING_DIR=$BIGWORK/$PROJECT_NAME
mkdir -p $WORKING_DIR
SLURMOUT_DIR={slurmout_dir}
mkdir -p $SLURMOUT_DIR  # must be created once beforehand :(

# show some information
python --version
echo "Python environment: $PYTHON_ENV_NAME"
echo "PATH: $PATH"
which python
conda info

# main command
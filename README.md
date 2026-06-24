# CINECA Leonardo HPC Project Setup

This guide provides instructions for connecting to the Leonardo cluster at CINECA, accessing the workspace, setting up the environment, and downloading models.

## 1. Connect to the Server

To access the Leonardo cluster, you need to authenticate and then SSH into the login node. Run the following commands from your local machine:

# Authenticate with your company email
step ssh login 'your.email@company.com' --provisioner cineca-hpc

# Connect to the login node (replace <your_username> with your actual username)
ssh <your_username>@login.leonardo.cineca.it


## 2. Access Your Workspace

Once logged in, navigate to the designated project directory in your workspace:

cd $WORK/project


## 3. Activate the Environment

Activate the Python virtual environment to ensure you have access to the required libraries:

source env/bin/activate

*Note on PyTorch Installation:*
PyTorch was installed in this environment using the following command (for CUDA 12.6 support):
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126


## 4. Test the PyTorch Installation

To verify that PyTorch is correctly configured and working, submit the test script to the queue:

sbatch submit_test_torch.sh 

Check the output to see your PyTorch version (replace XXX with the actual Job ID returned by sbatch):

tail -f slurm-XXX.out 


## 5. Download Hugging Face Models

You can download models directly from Hugging Face into your workspace. For example, to download the Qwen3-8B model:

hf download Qwen/Qwen3-8B --local-dir $WORK/project/Qwen3-8B

*If you need to download a different model, simply replace `Qwen/Qwen3-8B` and the target directory with the name of the desired model.*

#!/bin/bash
#SBATCH --account=AIFAC_F02_489           # Your project account
#SBATCH --partition=boost_usr_prod        # The GPU partition on Leonardo
#SBATCH --qos=boost_qos_lprod
#SBATCH --job-name=optuna_qwen            # Name of your job
#SBATCH --time=50:00:00                   # 12 hours for 5 sequential trials
#SBATCH --nodes=1                         # Single node
#SBATCH --ntasks-per-node=1               # Single task (no srun needed)
#SBATCH --cpus-per-task=8                 # Proportional CPU cores
#SBATCH --gres=gpu:1                      # Request 1x A100 GPU
#SBATCH --mem=64GB                        # Proportional Memory

# 1. Load the required system modules
module load python
module load cuda/12.3

# 2. Securely load your MLflow credentials into the environment
source $WORK/project/.mlflow_cr

# 3. Activate your Python virtual environment
source $WORK/project/env/bin/activate

# 4. Navigate to your working directory
cd $WORK/project

# 5. Run your fine-tuning script
python train_Qwen_8b_hyper_src.py


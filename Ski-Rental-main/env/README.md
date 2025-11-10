# 🧪 Environment Setup for Ski Rental Experiments

This folder contains the Conda environment configuration used for all experiments.

## 🔧 Installation Steps

1. Create the environment:
   ```bash
   conda env create -f env/environment.yml
   ```

2. Activate the environment:
   ```bash
   conda activate ski-rental
   ```

3. (Optional) If you run into PyMC-related warnings, check the troubleshooting section below.

## 🛠 Troubleshooting

- Make sure you're using `numpy==1.24` (PyMC compatibility)
- You may need to install compilers: `conda install gxx_linux-64` (Linux)

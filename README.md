\# MARL Environment Setup



\## Installation



This project uses Python packages listed in `requirements.txt`.



\### 1. Create and activate a virtual environment



Create a Python virtual environment and activate it before installing the dependencies.



For example:



```bash

python -m venv MARL

```



On Windows:



```bash

MARL\\Scripts\\activate

```



\### 2. Install the dependencies



Before installing the packages from `requirements.txt`, remove the PyTorch packages that have the `cu128` extension from the requirements file.



Install the remaining packages normally:



```bash

pip install -r requirements.txt

```



\### 3. Install the CUDA 12.8 versions of PyTorch



After the remaining dependencies have been installed, install the PyTorch packages separately using the CUDA 12.8 package index:



```bash

pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

```



This installs the CUDA 12.8 versions of PyTorch, torchvision, and torchaudio.



\### 4. Test the MARL environment



After installation is complete, run:



```bash

python -m test_marl_env.py

```



This script can be used to verify that the required environment and dependencies have been installed correctly.


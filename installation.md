
# Instruction Manual for Using TrackScoPy

## 1. Introduction

Welcome to TrackScoPy! This framework analyses fluorescence microscopy images, tracks microorganisms, and performs classification of the detected microorganisms from their trajectories. This manual will guide you through the installation and usage process to get you started. As a first step, download this directory (TrackScoPy) or clone this repository into your computer. 

## 2. System Requirements
- Operating System: Windows, macOS, or Linux
- Python 3.12.0 (Anaconda installation recommended but not mandatory, see instructions below)
- RAM: Minimum 4 GB (8 GB or more recommended)
- Disk Space: ~250 MB for installation

## 3. Installation

### Option A: Installing with Anaconda 

Anaconda is a free and open-source distribution of Python that simplifies package management and deployment. Follow these steps to install Anaconda:

1. **Download Anaconda**:  
   Visit the Anaconda download page:  
   https://www.anaconda.com/products/individual  
   Download the installer for your operating system (Windows, macOS, or Linux).

2. **Install Anaconda**:  
   Follow the installation instructions based on your operating system:  
   - **Windows**: Run the downloaded `.exe` file and follow the prompts. Make sure to check the box to "Add Anaconda to my PATH environment variable" during installation.  
   - **macOS/Linux**: Open a terminal and follow the instructions. You may need to use the command:  
     `bash [downloaded_file_name].sh`

3. **Verify Installation**:  
   Open a terminal (or Anaconda Prompt on Windows) and type:  
   `conda --version`  
   You should see the version number of Anaconda printed in the terminal.

4. **Install Required Packages**  
   Navigate to the TrackScoPy directory:

   ```
   cd [path_to_directory]
   ```

   Replace `[path_to_directory]` with the actual path to the TrackScoPy directory. If it includes spaces, enclose it in quotes, e.g., 

    ```
    cd "C:path\to\my directory"
    ```

   Create and activate a new environment, and install the dependencies:

   ```
   conda create -n trackscopy_env python=3.12 -y
   conda activate trackscopy_env
   pip install -r requirements.txt
   ```

Now, launch Jupyter Notebook (preferred), Jupyter Lab, or Visual Studio Code from the same terminal, e.g., type `jupyter notebook` and press enter.


### Option B: Installing Without Anaconda (Using `venv`)

1. **Make sure you have Python 3.12 and pip installed**. 

     Open a terminal (command prompt in Windows) and check using:

   ```
   python --version
   pip --version
   ```

2. **Create and activate a virtual environment**:

   - On Windows:

     ```
     python -m venv trackscopy_env
     trackscopy_env\Scripts\activate
     ```

   - On macOS/Linux:

     ```
     python3 -m venv trackscopy_env
     source trackscopy_env/bin/activate
     ```

3. **Install the dependencies**:

   ```
   pip install -r requirements.txt
   ```

Now, launch Jupyter Notebook (preferred), Jupyter Lab, or Visual Studio Code from the same terminal, e.g., type `jupyter notebook` and press enter.


## 4. Usage Instructions

After the initial setup, to use the framework later, open a terminal and activate the environment:

- If using Conda:

  ```
  conda activate trackscopy_env
  ```

- If using venv (on Windows):

  ```
  trackscopy_env\Scripts\activate
  ```

- If using venv (on macOS/Linux):

  ```
  source trackscopy_env/bin/activate
  ```

Then launch the Jupyter notebook:

```
jupyter notebook
```

You are now ready to run the framework.


You will find two `.ipynb` files and four sub-directories: `trackscopy_fluorescence`, `sample_data`, `sample_data_drift` and `website-pics` in addition to 'requirements.txt', 'LICENSE.txt', 'readme.md' and the instruction file 'installation.md'.

Inside Jupyter notebook, open the notebooks and follow the instructions in the corresponding files in the following order:

   - `1_main_image_analysis.ipynb`
   - `2_load_and_analyse.ipynb`

Inside `trackscopy_fluorescence`, the following `.py` files are included:

- `__init__.py`: ensures package recognition.
- `default_parameters_image_analysis.py`: image analysis parameters.
- `default_parameters_swim_mode_detection.py`: swim-mode detection parameters.
- `image_analysis_tools.py`: module containing image processing functions and classes.
- `swim_mode_analysis_tools.py`: module containing functions and classes corresponding to swim-mode detection of bacteria.

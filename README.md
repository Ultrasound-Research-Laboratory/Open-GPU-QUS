# GPU-QUS-GUI (Windows, GPU)

This repository provides two Windows-based graphical user interfaces (GUIs) for
quantitative ultrasound (QUS) processing using PyTorch CUDA and PySide6.

The tools are intended for **research use** and are optimized for **NVIDIA GPU**
execution.

---

## Citation

This code implements the methods described in the following paper:

Mingrui Liu, Michael L. Oelze. An open-source GPU-based real-time quantitative ultrasound toolbox. TechRxiv. February 06, 2026.
DOI: 10.36227/techrxiv.177042026.69094323/v1

If you use or adapt this code, please cite the above work.

---

## Included Applications

### 1) Attenuation Imaging (SLD)
- Backend: `apps/attenuation/GPU_SLD.py`
- GUI: `apps/attenuation/ui_app_paper.py`

Implements a GPU-accelerated spectral log-difference (SLD) attenuation pipeline.

---

### 2) Backscatter Coefficient (BSC)
- Backend: `apps/bsc/GPU_BSC.py`
- GUI: `apps/bsc/ui_app_bsc.py`

Implements GPU-accelerated backscatter coefficient (BSC) estimation.

---

## System Requirements

- **Operating System:** Windows 10 / Windows 11  
- **GPU:** NVIDIA GPU (CUDA-capable)  
- **Driver:** NVIDIA driver installed (CUDA toolkit is *not* required)

> CPU-only execution is **not officially supported** in this release.

---

## Option A: Run from Source (Developer Mode)

This option is recommended if you want to inspect or modify the code.

### 1) Create a Conda Environment

Open **Anaconda Prompt**:

```bat
conda create -n qus_gui_gpu python=3.10 -y
conda activate qus_gui_gpu
pip install -U pip
```

### 2) Install Dependencies
```bat
pip install pyside6 numpy scipy matplotlib
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3) Launch the GUIs
Attenuation (SLD):
```bat
cd apps\attenuation
python ui_app_paper.py
```

BSC:
```bat
cd apps\bsc
python ui_app_bsc.py
```

---
## Option B: Run from Prebuilt Release (Recommended)
For most users, download the **Windows GPU release ZIP** from the
**GitHub Releases** page.

### Steps
1. Download the ZIP file from GitHub Releases

2. Unzip it to a local folder (do not use a network drive)

3. Double-click:

launch_attenuation.bat

launch_bsc.bat

No Python or Conda installation is required on the target machine.

---

## Repository Structure

```text
GPU-QUS-GUI/
  apps/
    attenuation/
      GPU_SLD.py
      ui_app_paper.py
    bsc/
      GPU_BSC.py
      ui_app_bsc.py

  release_template/
    launch_attenuation.bat
    launch_bsc.bat
    README_release.txt

  scripts/
    make_release_gpu.ps1

  README.md
```

---
### Notes
The NVIDIA driver must be installed separately.

Output files are written to user-selected folders in the GUI.

This software is provided for research purposes and may be updated without
backward compatibility guarantees.

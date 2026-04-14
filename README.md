# **QPI-Deep-CWFS**

## **Description**
`QPI-Deep-CWFS` uses [OptiVolume](https://github.com/Muhammad-Kazim/OptiVolume) to create refractive index volumes with primitive shapes, such as cubes and ellipsoids, and simulates the imaging of the refractive index volume through the Coded Wavefront Sensing pipeline to create a Coded Wavefront Sensing dataset. This repository enables training/finetuning the RAFT optical flow network with synthetic data and comparing the QPI performance with conventional CWFS phase retrieval methods.

## **SynthEllips**
[SynthEllips](https://zenodo.org/records/18983874) is a coded wavefront sensing-based synthetic dataset consisting of 20.000 datapoints created using this repository. For each datapoint, a refractive index volume composed of a random configuration of ellipsoids (refractive indices, positions, diameters, and rotations) is imaged using a wave-optical simulation of the Coded Wavefront Sensing pipeline to create a reference-specimen speckle image pair. The amplitude and scaled gradient vector field in the phase mask are also provided to assist with the supervised training of optical flow neural networks to estimate the quantitative phase from a reference-specimen speckle image pair. Training with SynthEllips demonstrated strong generalization to real biological specimens recorded experimentally, and to optical systems with different diffusers/phase masks and microscopes. The QPI performance of these networks is found to be quantitatively and qualitatively superior to classical phase retrieval methods.
---

## **Features**
- **Custom CWFS dataset synthesis**: Change the configs/data_ceation.json to simulate different distributions of optical systems and create synthetic speckle image datasets.
- **Training and Evaluation**: Supervised training using RAFT and inference comparison with the [ADMM-based](https://github.com/Muhammad-Kazim/py_cwfs_alg) CWFS classical method.
---

## **Installation**

1. Create environment and download dependencies (OptiVolume package):
```bash
conda create --name env_name python=3.9
conda env update --name env_name -f https://raw.githubusercontent.com/Muhammad-Kazim/OptiVolume/main/arxiv/environment.yml
```

2. Clone the repository and update the environment:
```bash
git clone https://github.com/Muhammad-Kazim/qpi-deep-cwfs.git
cd qpi-deep-cwfs
conda env update --name env_name --file environment.yml
conda activate env_name
```

3. To create a dataset, update data_params.json in configs:
```bash
cd scripts
python 01_create_dataset.py --exp_name exp_name --data_params data_creation_cwfs_params
```

4. To train the network, update train_params.json in configs. RAFT with sample data in data/ can be trained using:
```bash
cd scripts
python 02_synthetic_RAFT_training_saved_dataset.py --exp_name temp_run --train_params train_params --dataset_path 2026-04-13_10_samples
```

5. To run inference, update infer_params.json in configs. Inference with examples ckpts can be done as follows:
```bash
cd scripts
python 03_synthetic_RAFT_training_saved_dataset_inference.py --infer_params_path infer_params
```

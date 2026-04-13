# **QPI-Deep-CWFS**

## **Description**
`QPI-Deep-CWFS` use OptiVolume to create refractive index volumes with primitive shapes, such as cubes and ellipsoids, and simulates the imaging of the refractive index volume through the Coded Wavefront Sensing pipleine to create a Coded Wavefront Sensing dataset. Optical-Flow neural networks finetuned on Coded Wavefront Sensing synthetic dataset generalize across biological specimen recorded experimentally, diffusers/phase masks, and microscopes. The QPI performance is quantiatively and qualitatively better than classical phase retrieval methods. 

---

## **Features**
- **Custom CWFS dataset synthesis**: Change the configs/data_ceation.json to simulate different distributions of optical systems and create synthetic speckle image datasets.
- **Training and Evaluation**: Supervised training using RAFT and inference comparison with ADMM-based CWFS classical method.
---

## **Installation**

1. Create enviroment and download dependencies (OptiVolume package):
```bash
conda env create --name env_name python=3.9
conda env update --name env_name -f https://raw.githubusercontent.com/Muhammad-Kazim/OptiVolume/main/arxiv/environment.yml
```

2. Clone repository and update environment:
```bash
git clone https://github.com/Muhammad-Kazim/qpi-deep-cwfs.git
cd qpi-deep-cwfs
conda env update --name env_name --file environment.yml
conda activate env_name
```

3. To create dataset, update data_params.json in configs:
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

# Cartesian Encoding Graph Neural Network for Crystal Structures Property Prediction: Application to Thermal Ellipsoid Estimation

![Pipeline](./fig/frontpage.png)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-cartesian-encoding-graph-neural-network-for/formation-energy-on-jarvis-dft-formation)](https://paperswithcode.com/sota/formation-energy-on-jarvis-dft-formation?p=a-cartesian-encoding-graph-neural-network-for) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-cartesian-encoding-graph-neural-network-for/band-gap-on-jarvis-dft)](https://paperswithcode.com/sota/band-gap-on-jarvis-dft?p=a-cartesian-encoding-graph-neural-network-for) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-cartesian-encoding-graph-neural-network-for/formation-energy-on-materials-project)](https://paperswithcode.com/sota/formation-energy-on-materials-project?p=a-cartesian-encoding-graph-neural-network-for) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-cartesian-encoding-graph-neural-network-for/band-gap-on-materials-project)](https://paperswithcode.com/sota/band-gap-on-materials-project?p=a-cartesian-encoding-graph-neural-network-for)



<h3 align="center">
  🌐 <a href="https://imatge-upc.github.io/CartNet/" target="_blank">Project</a> | 
  📃 <a href="https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00352g" target="_blank">Paper</a> |
  🤗 <a href="https://huggingface.co/spaces/alexsoleg/cartnet-demo" target="_blank">Demo</a>
</h3>




## Overview
CartNet is specifically designed for predicting Anisotropic Displacement Parameters (ADPs) in crystal structures. CartNet addresses the computational challenges of traditional methods by encoding the entire 3D geometry of atomic structures into a Cartesian reference frame instead of encoding based on only distance, bypassing the need for unit cell encoding. The model incorporates innovative features, including a neighbour equalization technique to enhance interaction detection and a Cholesky-based output layer to ensure valid ADP predictions. Additionally, it introduces a rotational SO(3) data augmentation technique to improve generalization across different crystal structure orientations, making the model highly efficient and accurate in predicting ADPs while significantly reducing computational costs.


Implementation of the CartNet model proposed in the paper:

- **Paper**: [Cartesian Encoding Graph Neural Network for Crystal Structures Property Prediction: Application to Thermal Ellipsoid Estimation](https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00352g)
- **Authors**: Àlex Solé, Albert Mosella-Montoro, Joan Cardona, Silvia Gómez-Coca, Daniel Aravena, Eliseo Ruiz and Javier Ruiz-Hidalgo
- **Journal**: [Digital Discovery](https://www.rsc.org/journals-books-databases/about-journals/digital-discovery/), 2025


## Table of Contents

- [Installation](#installation)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [Pre-trained Models](#pre-trained-models)
- [Known Issues](#known-issues)
- [Citation](#citation)
- [License](#license)



## Installation

Instructions to set up the environment:

```sh
# Clone the repository
git clone https://github.com/imatge-upc/CartNet.git
cd CartNet

# Create a Conda environment (original env)
conda env create -f environment.yml

# or alternatively, if you want to use torch 2.4.0
conda env create -f environment_2.yml

# Activate the environment
conda activate CartNet
```

## Dependencies

The environment used for the results reported in the paper relies on these dependencies:

```sh
pytorch==1.13.1
pytorch-cuda==11.7
pyg==2.5.2
pytorch-scatter==2.1.1
scikit-learn==1.5.1
scipy==1.13.1
pandas==2.2.2
wandb==0.17.3
yacs==0.1.6
jarvis-tools==2024.8.30
lightning==2.2.5
roma==1.5.0
e3nn==0.5.1
csd-python-api==3.3.1
```

These dependencies are automatically installed when you create the Conda environment using the `environment.yml` file.

### Update

We have updated our dependencies to torch 2.4.0 to facilitate further research. This can be installed via the `environment_2.yml` file.


## Dataset

### ADP Dataset:

The ADP (Anisotropic Displacement Parameters) dataset is curated from over 200,000 experimental crystal structures from the Cambridge Structural Database (CSD). This dataset is used to study atomic thermal vibrations represented through thermal ellipsoids. The dataset was curated to ensure high-quality and reliable ADPs. The dataset spans a wide temperature range (0K to 600K) and features a variety of atomic environments, with an average of 194.2 atoms per crystal structure. The dataset is split into 162,270 structures for training, 22,219 for validation, and 23,553 for testing.

The dataset can be generated using the following code:

```sh
cd dataset/
python extract_csd_data.py --output "/path/to/data/"
```

> [!NOTE]
> 
> Dataset generation requires a valid license for the [Cambridge Structural Database (CSD) Python API](https://downloads.ccdc.cam.ac.uk/documentation/API/index.html#).

### Jarvis
For tasks derived from Jarvis dataset, we followed the methodology of [Choudhary et al.](https://www.nature.com/articles/s41524-021-00650-1) in ALIGNN, utilizing the same training, validation, and test datasets. The dataset is automatically downloaded and processed by the code.

### The Materials Project

For tasks derived from The Materials Project, we followed the methodology of [Yan et al.](https://openreview.net/pdf?id=pqCT3L-BU9T) in Matformer, utilizing the same training, validation, and test datasets. The dataset is automatically downloaded and processed by the code, except for the bulk and shear modulus that are publicly available at [Figshare](https://figshare.com/projects/Bulk_and_shear_datasets/165430).

## Training

To recreate the experiments from the paper



### ADP:

To train **ADP Dataset** using **CartNet**:

```sh
cd scripts/
bash train_cartnet_adp.sh
```

To train **ADP Dataset** using **eComformer**:

```sh
cd scripts/
bash train_ecomformer_adp.sh
```
To train **ADP Dataset** using **eComformer**:

```sh
cd scripts/
bash train_icomformer_adp.sh
```

To run the ablation experiments in the **ADP Dataset**:

```sh
cd scripts/
bash run_ablations.sh
````

### Jarvis:

```sh
cd scripts/
bash train_cartnet_jarvis.sh
````

### The Materials Project

```sh
cd scripts/
bash train_cartnet_megnet.sh
```




## Evaluation

Instructions to evaluate the model:

```sh
python main.py --inference --checkpoint_path path/to/checkpoint.pth
```

<!-- TODO: table results -->
## Results

### ADP dataset

Results on ADP Dataset:

| Method      | MAE (Å²) ↓               | S₁₂ (%) ↓         | IoU (%) ↑          | #Params↓  |
|-------------|-------------------------|-------------------|-------------------|-----------|
| eComformer  | 6.22 · 10⁻³ ± 0.01 · 10⁻³ | 2.46 ± 0.01      | 74.22 ± 0.06     | 5.55M     |
| iComformer  | _3.22 · 10⁻³ ± 0.02 · 10⁻³_ | _0.91 ± 0.01_    | _81.92 ± 0.18_   | _4.9M_    |
| CartNet     | **2.87 · 10⁻³ ± 0.01 · 10⁻³** | **0.75 ± 0.01** | **83.56 ± 0.01** | **2.5M**  |

(best result in **bold** and second best in _italic_)

### Jarvis Dataset

Results on Jarvis Dataset:
| Method      | Form. Energy (meV/atom) ↓ | Band Gap (OPT) (meV) ↓ | Total energy (meV/atom) ↓ | Band Gap (MBJ) (meV) ↓ | Ehull (meV) ↓ |
|-------------|---------------------------|------------------------|---------------------------|------------------------|---------------|
| Matformer   | 32.5                      | 137                    | 35                        | 300                    | 64            |
| PotNet      | 29.4                      | 127                    | 32                        | 270                    | 55            |
| eComformer  | 28.4                      | 124                    | 32                        | 280                    | *44*          |
| iComformer  | *27.2*                    | *122*                  | *28.8*                    | *260*                  | 47            |
| CartNet     | **27.05 ± 0.07**         | **115.31 ± 3.36**     | **26.58 ± 0.28**         | **253.03 ± 5.20**      | **43.90 ± 0.36** |

(best result in **bold** and second best in _italic_)

### The Materials Project

| Method      | Form. Energy (meV/atom) ↓ | Band Gap (meV) ↓ | Bulk Moduli (log(GPa)) ↓ | Shear Moduli (log(GPa)) ↓ |
|-------------|---------------------------|------------------|--------------------------|--------------------------|
| Matformer   | 21                        | 211              | 0.043                    | 0.073                    |
| PotNet      | 18.8                      | 204              | 0.040                    | _0.065_                  |
| eComformer  | _18.16_                   | 202              | 0.0417                   | 0.0729                   |
| iComformer  | 18.26                     | _193_            | _0.038_                  | **0.0637**               |
| CartNet     | **17.47 ± 0.38**         | **190.79 ± 3.14** | **0.033 ± 0.00094**    | **0.0637 ± 0.0008**     |

(best result in **bold** and second best in _italic_)

## Pre-trained Models

Links to download pre-trained models:

- [CartNet ADP Dataset](https://zenodo.org/records/13970823?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImQxM2ExZjg2LWU4ODktNDhhZC04ODAxLTZjN2MxNGZjMWQ5ZSIsImRhdGEiOnt9LCJyYW5kb20iOiJhZWVlYjk2MmQ5ZjU1ODdiMDgzYmJhMDc4YWE1MTk3MyJ9.TLSJvG_khY3eD0bSWTGhZDtMS7YXk6KpXwLTXLDFPBbOF4PhlxlAeL9h2rpb3M20ushHhqOSfwfvutdfjAePhw
)


## Known Issues

Due to the presence of certain non-deterministic operations in PyTorch, as discussed [here](https://pytorch.org/docs/stable/notes/randomness.html), some results may not be exactly reproducible and may exhibit slight variations. This variability can also arise when using different GPU models for training and testing the network.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{solé2025cartesianencodinggraphneural,
      title={A Cartesian Encoding Graph Neural Network for Crystal Structures Property Prediction: Application to Thermal Ellipsoid Estimation}, 
      author={Àlex Solé and Albert Mosella-Montoro and Joan Cardona and Silvia Gómez-Coca and Daniel Aravena and Eliseo Ruiz and Javier Ruiz-Hidalgo},
      year={2025},
      eprint={2501.18369},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.18369}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions and/or suggestions please contact [jaume.alexandre.sole@upc.edu](mailto:jaume.alexandre.sole@upc.edu)





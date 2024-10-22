# CartNet: Cartesian Encoding for Anisotropic Displacement Parameters Estimation

![Pipeline](./fig/pipeline.png)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


## Overview
CartNet is specifically designed for predicting Anisotropic Displacement Parameters (ADPs) in crystal structures. CartNet addresses the computational challenges of traditional methods by encoding the full 3D geometry of atomic structures into a Cartesian reference frame, bypassing the need for unit cell encoding. The model incorporates innovative features, including a neighbour equalization technique to enhance interaction detection and a Cholesky-based output layer to ensure valid ADP predictions. Additionally, it introduces a rotational SO(3) data augmentation technique to improve generalization across different crystal structure orientations, making the model highly efficient and accurate in predicting ADPs while significantly reducing computational costs


Implementation of the CartNet model proposed in the paper:

- **Paper**: [CartNet: Cartesian Encoding for Anisotropic Displacement Parameters Estimation](link_to_paper)
- **Authors**: Àlex Solé, Albert Mosella-Montoro, Joan Cardona, Silvia Gómez-Coca, Daniel Aravena, Eliseo Ruiz and Javier Ruiz-Hidalgo
- **Conference/Journal**: [Digital Discovery](https://www.rsc.org/journals-books-databases/about-journals/digital-discovery/), Year

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

# Create a Conda environment
conda env create -f environment.yml

# Activate the environment
conda activate CartNet
```

## Dependencies

The environment relies on these dependencies:

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
```

These dependencies are automatically installed when you create the Conda environment using the `environment.yml` file.


## Dataset

### ADP Dataset:

The ADP (Anisotropic Displacement Parameters) dataset is curated from over 200,000 experimental crystal structures from the Cambridge Structural Database (CSD). This dataset is used to study atomic thermal vibrations represented through thermal ellipsoids. The dataset was curated to ensure high-quality and reliable ADPs. The dataset spans a wide temperature range (0K to 600K) and features a variety of atomic environments, with an average of 194.2 atoms per crystal structure. The dataset is split into 162,270 structures for training, 22,219 for validation, and 23,553 for testing.

The ADP dataset can be downloaded from the following [link](https://drive.google.com/file/d/1lCKGG0Jtd7tjxmj1TgPkN8D_8uXXKlKF/view?usp=sharing).

The dataset can be extracted using:

```bash
tar -xvzf adp_dataset.tar.gz
```

> [!NOTE]
>
> The ADP_DATASET/ folder should be placed inside the dataset/ folder or specify the new path via --dataset_path flag in main.py

### Jarvis
For tasks derived from Jarvis datset, we folowed the methdology of [Choudhary et al.](https://www.nature.com/articles/s41524-021-00650-1) in ALIGNN, utilizing the same training, validation, and test datasets. The dataset is automatically downloaded and processed by the code.

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

| Method      | MAE (Å²) ↓        | S₁₂ (%) ↓       | IoU (%) ↑       | #Params  |
|-------------|-------------------|-----------------|-----------------|----------|
| eComformer  | 6.21 · 10⁻³       | 2.47            | 74.34           | 5.55M    |
| iComformer  | _3.22 · 10⁻³_     | _0.94_          | _81.97_         | 4.9M     |
| CartNet     | **2.88 · 10⁻³**   | **0.75**        | **83.53**       | 2.5M     |

(best result in **bold** and second best in _italic_)

### Jarvis Dataset

Results on Jarvis Dataset:
| Method      | Form. Energy (meV/atom) ↓ | Band Gap (OPT) (meV) ↓ | Total energy (meV/atom) ↓ | Band Gap (MBJ) (meV) ↓ | Ehull (meV) ↓ |
|-------------|---------------------------|------------------------|---------------------------|------------------------|---------------|
| Matformer   | 32.5                      | 137                    | 35                        | 300                    | 64            |
| PotNet      | 29.4                      | 127                    | 32                        | 270                    | 55            |
| eComformer  | 28.4                      | 124                    | 32                        | 280                    | *44*          |
| iComformer  | *27.2*                    | *122*                  | *28.8*                    | *260*                  | 47            |
| CartNet     | **27.03**                 | **111.2**              | **26.26**                 | **247**                | **43.54**     |

(best result in **bold** and second best in _italic_)

### The Materials Project

| Method      | Form. Energy (meV/atom) ↓ | Band Gap (meV) ↓ | Bulk Moduli (log(GPa)) ↓ | Shear Moduli (log(GPa)) ↓ |
|-------------|---------------------------|------------------|--------------------------|--------------------------|
| Matformer   | 21                        | 211              | 0.043                    | 0.073                    |
| PotNet      | 18.8                      | 204              | 0.04                     | 0.065                    |
| eComformer  | *18.16*                   | 202              | 0.0417                   | 0.0729                   |
| iComformer  | 18.26                     | *193*            | *0.038*                  | *0.0637*                 |
| CartNet     | **16.95**                 | **186**          | **0.0321**               | **0.0628**               |

(best result in **bold** and second best in _italic_)

## Pre-trained Models

Links to download pre-trained models:

- [CartNet ADP Dataset](https://zenodo.org/records/13970823?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImQxM2ExZjg2LWU4ODktNDhhZC04ODAxLTZjN2MxNGZjMWQ5ZSIsImRhdGEiOnt9LCJyYW5kb20iOiJhZWVlYjk2MmQ5ZjU1ODdiMDgzYmJhMDc4YWE1MTk3MyJ9.TLSJvG_khY3eD0bSWTGhZDtMS7YXk6KpXwLTXLDFPBbOF4PhlxlAeL9h2rpb3M20ushHhqOSfwfvutdfjAePhw
)


## Known Issues

Due to the presence of certain non-deterministic operations in PyTorch, as discussed [here](https://pytorch.org/docs/stable/notes/randomness.html), some results may not be fully reproducible or may exhibit slight variations. This variability can also arise when using different GPU models for training and testing the network.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper_citation,
  title={Title of the Paper},
  author={Author1 and Author2 and Author3},
  journal={Journal Name},
  year={2023},
  volume={XX},
  number={YY},
  pages={ZZZ}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions and/or suggestions please contact [jaume.alexandre.sole@upc.edu](mailto:jaume.alexandre.sole@upc.edu)





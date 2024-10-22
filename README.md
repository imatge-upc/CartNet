# CartNet: Cartesian Encoding for Anisotropic Displacement Parameters Estimation

![Pipeline](./fig/pipeline.png)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


## Overview

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
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)


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

The ADP dataset can be downloaded from the following link.

The dataset can be extracted using:
tar -xf adp_dataset.tar.gz

[!NOTE]

The ADP_DATASET/ folder should be placed inside the dataset/ folder or scpecify the new path via --dataset_path flag in main.py




## Training

To recreate the experiments from the paper:


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
# Command to evaluate the model
python main.py --inference --checkpoint_path path/to/checkpoint.pth
```

## Results

Include quantitative results, such as accuracy, and qualitative results, like sample outputs:

| Metric   | Value |
| -------- | ----- |
| Accuracy | 95%   |
| F1 Score | 0.94  |
| ...      | ...   |

## Pre-trained Models

Links to download pre-trained models:

- [Model Name](link_to_model) (e.g., Google Drive, AWS S3)


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

## Acknowledgments

- Mention any collaborators or funding sources.
- Credit libraries or resources that were helpful.





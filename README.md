# MICLe_Pytorch
Unoffitial PyTorch implementation of MICLe implementation of the paper [Azizi, S., Mustafa, B., Ryan, F., Beaver, Z., Freyberg, J., Deaton, J., ... & Norouzi, M. (2021). Big self-supervised models advance medical image classification. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 3478-3488).](https://arxiv.org/abs/2101.05224)

## Requirements
- Python 3.7
- Pytorch 1.10.*
- Pytorch-ligntning 1.6.*
- lightly 1.2.*
- Pillow 8.4.*

## Usage

### 1. Prepare the dataset

You can arrange your dataset in the following structure:

```bash
├── dataset
│   ├── subject 1
│   │   ├── image 1
│   │   ├── image 2
│   │   ├── ...
│   ├── subject 2
│   │   ├── image 1
│   │   ├── image 2
│   │   ├── ...
│   ├── ...
```

To train, run the following command with the path to your dataset:

```bash
python main.py --train_dir_path /your/path/train_dataset
```

You can also specify the hyperparameters in the command line.

The model will be saved as a .pth in your project directory.

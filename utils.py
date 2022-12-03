import re
import torch
from tqdm import tqdm


def regular_exp(pattern, txt):

    matched_pattern = re.findall(pattern, txt)
    return matched_pattern[0]


def cal_mean_std(dataloader):
    """
    take img_dataloader as input and
    calculate the mean and std of the images.
    :param dataloader
    :return: mean and std of all image tensor in dataloader
    """
    nimages = 0.0
    mean = 0.0
    var = 0.0

    for index, batch in tqdm(enumerate(dataloader)):
        img_batch = batch[0]
        img_batch = img_batch.view(img_batch.size(0),
                                   img_batch.size(1),
                                   -1)

        nimages += img_batch.size(0)
        mean1 = img_batch.mean(2)
        mean += img_batch.mean(2).sum(0)
        var += img_batch.var(2).sum(0)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)

    return mean, std
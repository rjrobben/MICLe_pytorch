import os
import random
import torch.utils.data as data
from PIL import Image
from utils import regular_exp


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class MICLeDataset(data.Dataset):

    def __init__(self,
                 img_dir,
                 transform=None,
                 augmentation = None):
        """
        :param img_dir: Path to the folder with all subdirs, each subdir contains images of one subject.
        |--- img_dir
        |    |--- subject 1
        |    |    |--- 1_1.jpg
        |    |    |--- 1_2.jpg
        |    |    |--- 1_3.jpg
        |    |--- subject 2
        |    |    |--- 2_1.jpg
        |    |    |--- 2_2.jpg
        |    |    |--- 2_3.jpg
        |    |--- subject 3 
        |    |    |--- 3_1.jpg
        |    |    |--- 3_2.jpg
        |    |    |--- 3_3.jpg
        :param transform: transform to be applied on the image
        :param augmentation: augmentation to be applied on the image
        """

        super().__init__()

        self.img_dir = img_dir
        self.folder_list = os.listdir(img_dir)
        self.transform = transform

        self.augmentation = augmentation

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        folder_name = self.folder_list[idx]
        folder_path = os.path.join(self.img_dir, folder_name)
        fname_list = os.listdir(folder_path)
        if len(fname_list) >=2:
            two_random_imgs = random.sample(fname_list, 2)
            img_path1 = os.path.join(folder_path, two_random_imgs[0])
            img_path2 = os.path.join(folder_path, two_random_imgs[1])
            image1 = pil_loader(img_path1)
            image2 = pil_loader(img_path2)

            if self.transform:
                image1 = self.transform(image1)
                image2 = self.transform(image2)
            return image1, image2, two_random_imgs[0] + "&" + two_random_imgs[1]

        elif len(fname_list) == 1:
            img_path = os.path.join(folder_path, fname_list[0])
            image = pil_loader(img_path)

            if self.augmentation:
                imageA = self.augmentation(image)
                imageB = self.augmentation(image)
            else:
                imageA = None
                imageB = None

            return imageA, imageB, fname_list[0]
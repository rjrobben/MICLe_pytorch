from MICLe import MICLeModel
from collate import SimCLRCollateFunction
import pytorch_lightning as pl
import torch
import argparse
from dataset import MICLeDataset
import torchvision

def main(train_dir_path, 
         epochs=100,
         seed=1,
         devices=2,
         accelerator='gpu',
         img_size=256,
         batch_size=256,
         num_workers=8,
         lr=0.1,
         momentum=0.9,
         weight_decay=1e-5,
         trust_coefficient=0.001,
         temperature=0.1,
         model_output_name='model_output.pth'):

    class Hparams:
        def __init__(self):
            self.epochs = epochs
            self.seed = seed
            self.devices = [devices]
            self.accelerator = accelerator
            self.IMG_SIZE_X = img_size
            self.IMG_size_Y = img_size
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.lr = lr
            self.momentum = momentum
            self.weight_decay = weight_decay
            self.trust_coefficient = trust_coefficient
            self.temperature = temperature
            self.model_output_name = model_output_name

    train_config = Hparams()

    collate_fn = SimCLRCollateFunction(
                    input_size=train_config.IMG_SIZE_X,
                    vf_prob=0.5,
                    rr_prob=0.5,
                    hf_prob=0.5,
                    gaussian_blur=0.,
                        normalize={
                            'mean':[0.4561, 0.2746, 0.1623],
                            'std': [0.2609, 0.1639, 0.0991]
                    })

    transform = torchvision.transforms.Compose([
        torchvision.Resize([train_config.IMG_SIZE_X, train_config.IMG_SIZE_Y]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.4561, 0.2746, 0.1623), std=(0.2609, 0.1639, 0.0991))
    ])

    color_jitter = torchvision.transforms.ColorJitter(
        0.7, 0.7, 0.7, 0.2
    )

    augment = torchvision.transforms.Compose([
        torchvision.transforms.RandomApply([color_jitter], p=0.8),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.ToTensor()

    ])

    
    fundus_img_data = MICLeDataset(img_dir=train_dir_path,transform=transform,augmentation=augment)
    
    dataloader_train_MICLe = torch.utils.data.DataLoader(
                                fundus_img_data,
                                batch_size=train_config.batch_size,
                                shuffle=True,
                                # collate_fn=collate_fn,
                                drop_last=True,
                                num_workers=train_config.num_workers)

    pl.seed_everything(train_config.seed)
    model = MICLeModel(config=train_config)

    trainer = pl.Trainer(
        auto_lr_find=False,
        max_epochs=train_config.epochs,
        devices=train_config.devices,
        accelerator=train_config.accelerator,
        progress_bar_refresh_rate=1
    )
    trainer.fit(model, dataloader_train_MICLe)

    pretrained_resnet_backbone = model.backbone
    state_dict = {
        'resnet18_parameters': pretrained_resnet_backbone.state_dict()
    }
    torch.save(state_dict, train_config.model_output_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # optimization related arguments
    parser.add_argument('--train_dir_path', type=str,
                        help='path to your training directory')
    parser.add_argument('--epochs', default=100, type=int,
                        help='epochs to train for')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--devices', default=2, help='device index to specify device to train on')
    parser.add_argument('--accelerator', default='gpu', type = str, help='device to train on (cpu, gpu, ddp)')
    parser.add_argument('--img_size', default=256, type = int, help='image size')
    parser.add_argument('--batch_size', default=256, type = int, help='batch size')
    parser.add_argument('--lr', default=0.1, type=float, help='LR')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=.1e-5, type=float)
    parser.add_argument('--trust_coefficient', default=0.001, type=float, help='trust coefficient')
    parser.add_argument('--temperature', default=0.1, type=float, help='temperature')
    parser.add_argument('--model_output_name', default="model_output.pth", type=str, help='name of the output model file')
    opt = parser.parse_args()

    main(**vars(opt))
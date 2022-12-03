import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
from pl_bolts.optimizers.lars import LARS


class MICLeModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters('config')
 
        self.config = config 
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18(pretrained=True)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss(temperature=0.1)

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        x0, x1, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        max_epochs = int(self.config.epochs)
        lr = self.config.lr  # 0.1

        optim = LARS(self.parameters(),
                     lr=lr,
                     momentum=self.config.momentum,
                     weight_decay=self.config.weight_decay,
                     trust_coefficient=self.config.trust_coefficient)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            max_epochs
        )
        return [optim], [scheduler]
import torch.nn as nn

from uas_mood.models.feature_extractor import Extractor


class FeatureAE(nn.Module):
    def __init__(self, c_z, img_size, ks=3, use_batchnorm=True,
                 backbone="resnet18", cnn_layers=["layer1", "layer2", "layer3"]):
        super().__init__()

        self.extractor = Extractor(
            backbone=backbone,
            cnn_layers=cnn_layers,
            img_size=img_size,
            upsample="nearest",
            is_agg=False,
            featmap_size=img_size,
            keep_feature_prop=(200 / 448)
        )
        c_in = self.extractor.c_out

        pad = ks // 2

        # Encoder
        enc = []
        # Layer 1
        enc += [nn.Conv2d(c_in, (c_in + 2 * c_z) // 2, kernel_size=ks,
                          padding=pad, bias=False)]
        if use_batchnorm:
            enc += [nn.BatchNorm2d((c_in + 2 * c_z) // 2)]
        enc += [nn.LeakyReLU()]
        # Layer 2
        enc += [nn.Conv2d((c_in + 2 * c_z) // 2, 4 * c_z, kernel_size=ks,
                          padding=pad, bias=False)]
        if use_batchnorm:
            enc += [nn.BatchNorm2d(4 * c_z)]
        enc += [nn.LeakyReLU()]
        # Layer 2.1
        # ---------------------------------------------------------------
        enc += [nn.Conv2d(4 * c_z, 2 * c_z, kernel_size=ks,
                          padding=pad, bias=False)]
        if use_batchnorm:
            enc += [nn.BatchNorm2d(2 * c_z)]
        enc += [nn.LeakyReLU()]
        # ---------------------------------------------------------------
        # Layer 3
        enc += [nn.Conv2d(2 * c_z, c_z, kernel_size=ks, padding=pad,
                          bias=False)]
        self.encoder = nn.Sequential(*enc)

        # Decoder
        dec = []
        # Layer 1
        dec += [nn.Conv2d(c_z, 2 * c_z, kernel_size=ks, padding=pad,
                          bias=False)]
        if use_batchnorm:
            dec += [nn.BatchNorm2d(2 * c_z)]
        dec += [nn.LeakyReLU()]
        # Layer 2.1
        # ---------------------------------------------------------------
        dec += [nn.Conv2d(2 * c_z, 4 * c_z, kernel_size=ks, padding=pad,
                          bias=False)]
        if use_batchnorm:
            dec += [nn.BatchNorm2d(4 * c_z)]
        dec += [nn.LeakyReLU()]
        # ---------------------------------------------------------------
        # Layer 2
        dec += [nn.Conv2d(4 * c_z, (c_in + 2 * c_z) // 2, kernel_size=ks,
                          padding=pad, bias=False)]
        if use_batchnorm:
            dec += [nn.BatchNorm2d((c_in + 2 * c_z) // 2)]
        dec += [nn.LeakyReLU()]
        # Layer 3
        dec += [nn.Conv2d((c_in + 2 * c_z) // 2, c_in, kernel_size=ks,
                          padding=pad, bias=False)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        feats = self.extractor(x)
        z = self.encoder(feats)
        feats_pred = self.decoder(z)
        return feats, feats_pred

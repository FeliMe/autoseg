from collections import OrderedDict
from math import log
from typing import List

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


""""""""""""""""""""""""""""""""" Utilities """""""""""""""""""""""""""""""""


def weights_init_leaky_relu(m):
    # Initializing ConvTranspose2d as Kaiming had negative effects
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(
            m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(
            m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def weights_init_relu(m):
    # Initializing ConvTranspose2d as Kaiming had negative effects
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def weights_init_relu_normal(m):
    # Initializing ConvTranspose2d as Kaiming had negative effects
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


""""""""""""""""""""""""""""""""" Wide ResNet AE """""""""""""""""""""""""""""""""


def batchnorm(in_channels: int):
    return nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.9)


def conv2d(in_channels: int, out_channels: int, kernel_size: int,
           stride: int = 1):
    pad = kernel_size // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                     padding=pad, bias=False)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 dropout_rate: float = 0.0):
        super().__init__()
        self.is_channels_equal = in_channels == out_channels
        self.stride = stride

        self.bn1 = batchnorm(in_channels)
        self.conv1 = conv2d(in_channels, out_channels, 3, 1)
        self.bn2 = batchnorm(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv2 = conv2d(out_channels, out_channels, 3, 1)

        # Shortcut needs conv if in_channels != out_channels
        if not self.is_channels_equal:
            self.shortcut_conv = conv2d(in_channels, out_channels, 1, 1)

    @staticmethod
    def _upsample(inp: torch.Tensor, scale_factor: int):
        return F.interpolate(inp, scale_factor=scale_factor, mode="bilinear",
                             align_corners=True)

    def forward(self, x: torch.Tensor):

        # Main path
        y = F.relu(self.bn1(x))  # BatchNorm + ReLU
        y = self.conv1(y)  # First convolution
        y = self._upsample(y, self.stride)  # Bilinear upsample
        y = F.relu(self.bn2(y))  # BatchNorm + ReLU
        y = self.dropout(y)  # Dropout
        y = self.conv2(y)  # Second convolution

        # Residual path
        if self.is_channels_equal and self.stride == 1:
            shortcut = x
        elif self.is_channels_equal and self.stride == 2:
            shortcut = self._upsample(x, self.stride)
        else:
            shortcut = self.shortcut_conv(x)
            shortcut = self._upsample(shortcut, self.stride)

        y = y + shortcut

        return y


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 dropout_rate: float = 0.0):
        super().__init__()
        self.is_channels_equal = in_channels == out_channels
        self.stride = stride

        self.bn1 = batchnorm(in_channels)
        self.conv1 = conv2d(in_channels, out_channels, 3, stride)
        self.bn2 = batchnorm(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv2 = conv2d(out_channels, out_channels, 3, 1)

        # Shortcut needs conv if in_channels != out_channels
        if not self.is_channels_equal:
            self.shortcut_conv = conv2d(in_channels, out_channels, 1, stride)
        if self.is_channels_equal and self.stride == 2:
            self.pool = nn.AvgPool2d(stride, stride)

    def forward(self, x: torch.Tensor):

        # Main path
        y = F.relu(self.bn1(x))  # BatchNorm + ReLU
        y = self.conv1(y)  # First convolution
        y = F.relu(self.bn2(y))  # BatchNorm + ReLU
        y = self.dropout(y)  # Dropout
        y = self.conv2(y)  # Second convolution

        # Residual path
        if self.is_channels_equal and self.stride == 1:
            shortcut = x
        elif self.is_channels_equal and self.stride == 2:
            shortcut = self.pool(x)
        else:
            shortcut = self.shortcut_conv(x)

        y = y + shortcut

        return y


def conv_group(in_channels: int, out_channels: int, n: int,
               stride: int, dropout_rate: float = 0.0):
    layers = [BasicBlock(in_channels, out_channels, stride, dropout_rate)]
    for _ in range(1, n):
        layers += [BasicBlock(out_channels, out_channels, 1, dropout_rate)]
    return nn.Sequential(*layers)


def upsample_group(in_channels: int, out_channels: int, n: int,
                   stride: int, dropout_rate: float = 0.0):
    layers = [UpsampleBlock(in_channels, out_channels, stride, dropout_rate)]
    for _ in range(1, n):
        layers += [UpsampleBlock(out_channels, out_channels, 1, dropout_rate)]
    return nn.Sequential(*layers)


class WideResNetAE(nn.Module):
    def __init__(self, inp_size: int, widen_factor: int = 1,
                 dropout_rate: float = 0.0):
        """Wide ResNet Autoencoder. Works for image sizes of 128, 256, and 512"""
        super().__init__()

        assert (int(log(inp_size, 2)) - log(inp_size, 2)
                == 0), "inp_size must be a power of 2"

        channels = [
            16,
            16 * widen_factor,
            32 * widen_factor,
            64 * widen_factor,
            64 * widen_factor,
            64 * widen_factor,
            64 * widen_factor
        ]

        # Encoder
        enc = [
            conv2d(1, channels[0], 3),
            conv_group(channels[0], channels[1], 1, 1,
                       dropout_rate),  # [b, 16*w, 256, 256]
            conv_group(channels[1], channels[2], 1, 2,
                       dropout_rate),  # [b, 32*w, 128, 128]
            conv_group(channels[2], channels[3], 1, 2,
                       dropout_rate),  # [b, 64*w, 64, 64]
            conv_group(channels[3], channels[4], 1, 2,
                       dropout_rate),  # [b, 64*w, 32, 32]
        ]
        if inp_size > 128:
            enc += [conv_group(channels[4], channels[5], 1, 2, dropout_rate)]
        if inp_size > 256:
            enc += [conv_group(channels[5], channels[6], 1, 2, dropout_rate)]
        self.enc = nn.Sequential(*enc)

        # Decoder
        dec = []
        if inp_size > 256:
            dec += [upsample_group(channels[6],
                                   channels[2], 1, 2, dropout_rate)]
            dec += [upsample_group(channels[2],
                                   channels[2], 1, 2, dropout_rate)]
            dec += [upsample_group(channels[2],
                                   channels[1], 1, 2, dropout_rate)]
        elif inp_size > 128:
            dec += [upsample_group(channels[5],
                                   channels[2], 1, 2, dropout_rate)]
            dec += [upsample_group(channels[2],
                                   channels[1], 1, 2, dropout_rate)]
        else:
            dec += [upsample_group(channels[4],
                                   channels[1], 1, 2, dropout_rate)]
        dec += [
            upsample_group(channels[1], channels[0], 1, 2, dropout_rate),
            upsample_group(channels[0], 1, 1, 2, dropout_rate)
        ]
        self.dec = nn.Sequential(*dec)

        # Weight init
        self.apply(weights_init_relu)

    def forward(self, x: torch.Tensor):

        z = self.enc(x)
        y = self.dec(z)

        # Final activation
        y = torch.sigmoid(y)

        return y


""""""""""""""""""""""""""""""""" UNet """""""""""""""""""""""""""""""""


class UNet(nn.Module):
    """From https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py"""
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


""""""""""""""""""""""""""""""""" UNet """""""""""""""""""""""""""""""""


class VAE(nn.Module):
    """Source: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py"""
    def __init__(self,
                 img_size: Tensor,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 num_layers: int = 5,
                 latent_dim: int = 256,
                 model_width: int = 32,
                 hidden_dims: List = None) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        assert len(img_size) == 2
        if isinstance(img_size, list) or isinstance(img_size, tuple):
            img_size = torch.tensor(img_size)

        if hidden_dims is None:
            hidden_dims = [model_width * (2**i) for i in range(num_layers)]
        self.hidden_dims = hidden_dims

        modules = []

        """ Build encoder """
        self.feat_size = img_size
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
            self.feat_size = torch.div(self.feat_size, 2, rounding_mode="trunc")  # floor divide

        self.encoder = nn.Sequential(*modules)
        n_feats = int(torch.prod(self.feat_size)) * hidden_dims[-1]
        self.fc_mu = nn.Linear(n_feats, latent_dim)
        self.fc_var = nn.Linear(n_feats, latent_dim)


        """ Build decoder """
        modules = []

        self.decoder_input = nn.Linear(latent_dim, n_feats)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1,
                                       bias=False),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=4,
                                               stride=2,
                                               padding=1,
                                               bias=False),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels,
                                      kernel_size=3, padding=1),
                            nn.Tanh())

    def encode(self, inp: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param inp: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(inp)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[0], *self.feat_size.tolist())
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, inp: Tensor) -> List[Tensor]:
        mu, log_var = self.encode(inp)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return [reconstruction, mu, log_var]

    @staticmethod
    def loss_function(reconstruction: Tensor,
                      inp: Tensor,
                      mu: Tensor,
                      log_var: Tensor,
                      M_N: float) -> list:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param reconstruction: (Tensor)
        :param inp: (Tensor)
        :param mu: (Tensor)
        :param log_var: (Tensor)
        :param M_N: (float)
        :return:
        """
        # Account for the minibatch samples from the dataset
        # M_N = batch_size / len(dataset)
        kld_weight = M_N
        recons_loss = F.mse_loss(reconstruction, inp)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return [loss, recons_loss, kld_loss]


""""""""""""""""""""""""""""""""" ACSUNet """""""""""""""""""""""""""""""""

class ACSUNet(nn.Module):
    # conv = ACSConv
    conv = nn.Conv3d
    def __init__(self, in_channels=3, out_channels=1, init_features=16):
        super(ACSUNet, self).__init__()

        features = init_features

        # Encoder
        self.encoder1 = ACSUNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = ACSUNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ACSUNet._block(features * 2, features * 2, name="bottleneck")

        # Decoder
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.decoder2 = ACSUNet._block((features * 2) * 2, features * 1, name="dec2")
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.decoder1 = ACSUNet._block(features * 2, features, name="dec1")

        self.final_conv = ACSUNet.conv(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        # Encode
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))

        # Decode
        dec2 = self.up2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.final_conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        ACSUNet.conv(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        ACSUNet.conv(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


if __name__ == '__main__':
    device = "cpu"
    # size = 256
    # model = UNet(in_channels=1, out_channels=1, init_features=32).to(device)
    # x = torch.randn(2, 1, size, size).to(device)
    # y = model(x)

    size = 256
    model = WideResNetAE(inp_size=size).to(device)
    x = torch.randn(2, 1, size, size).to(device)
    y = model(x)

    # size = 256
    # batch_size = 2
    # model = VAE(in_channels=1, img_size=torch.tensor([size, size])).to(device)
    # x = torch.randn(batch_size, 1, size, size).to(device)
    # print(summary(model, input_size=x.shape[1:], batch_size=2, device=device))
    # y = model(x)

    import IPython ; IPython.embed() ; exit(1)

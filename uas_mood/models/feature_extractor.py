import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models as tv_models


RESNETLAYERS = ["layer0", "layer1", "layer2", "layer3", "layer4", "avgpool"]


def _set_requires_grad_false(layer):
    for param in layer.parameters():
        param.requires_grad = False


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, resnet, layer_names=RESNETLAYERS):
        """
        Returns features on multiple levels from a ResNet18.

        Available layers: 'layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool'

        Args:
            resnet (nn.Module): Type of resnet used
            layer_names (list): List of string of layer names where to return
                                the features. Must be ordered
        Returns:
            out (dict): Dictionary containing the extracted features as
                        torch.tensors
        """
        super().__init__()

        _set_requires_grad_false(resnet)

        # [b, 3, 256, 256]
        self.layer0 = nn.Sequential(
            *list(resnet.children())[:4])  # [b, 64, 64, 64]
        self.layer1 = resnet.layer1  # [b, 64, 64, 64]
        self.layer2 = resnet.layer2  # [b, 128, 32, 32]
        self.layer3 = resnet.layer3  # [b, 256, 16, 16]
        self.layer4 = resnet.layer4  # [b, 512, 8, 8]
        self.avgpool = resnet.avgpool  # [b, 512, 1, 1]

        self.layer_names = layer_names

    def forward(self, inp):
        # Add fake batch dimension if necessary
        if inp.ndim == 3:
            inp = inp.unsqueeze(0)

        # Repeat channel dimension if grayscale
        if inp.shape[1] == 1:
            inp = inp.repeat(1, 3, 1, 1)
            # inp[:, 1] *= -1  # Invert one channel

        out = {}
        for name, module in self._modules.items():
            inp = module(inp)
            if name in self.layer_names:
                out[name] = inp
            if name == self.layer_names[-1]:
                break
        return out


class ResNet18FeatureExtractor(ResNetFeatureExtractor):
    def __init__(self, layer_names=RESNETLAYERS):
        super().__init__(tv_models.resnet18(pretrained=True), layer_names)


class WideResNet50FeatureExtractor(ResNetFeatureExtractor):
    def __init__(self, layer_names=RESNETLAYERS):
        super().__init__(
            torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2',
                           pretrained=True),
            layer_names
        )


class Extractor(nn.Module):
    """
    Muti-scale regional feature extractor
    """
    BACKBONENETS = {
        "resnet18": ResNet18FeatureExtractor,
        "wide_resnet50": WideResNet50FeatureExtractor
    }

    def __init__(
        self,
        backbone="resnet18",
        cnn_layers=RESNETLAYERS,
        upsample="nearest",
        featmap_size=256,  # Before convolving
        img_size=256,
        keep_feature_prop=1.0,
    ):
        super().__init__()

        self.backbone = self.BACKBONENETS[backbone](layer_names=cnn_layers)
        self.featmap_size = featmap_size
        self.img_size = img_size
        self.upsample = upsample
        self.align_corners = True if upsample == "bilinear" else None

        # Find out how many channels we got from the backbone
        c_out = self._get_out_channels()

        # Create mask to drop random features_channels
        self.feature_mask = torch.Tensor(c_out).uniform_() < keep_feature_prop
        self.c_out = self.feature_mask.sum().item()

    def _get_out_channels(self):
        device = next(self.backbone.parameters()).device
        inp = torch.randn((2, 1, self.img_size, self.img_size), device=device)
        feat_maps = self.backbone(inp)
        channels = 0
        for feat_map in feat_maps.values():
            channels += feat_map.shape[1]
        return channels

    def forward(self, inp):
        if isinstance(inp, dict):
            feat_maps = inp
        else:
            feat_maps = self.backbone(inp)

        features = []
        for _, feat_map in feat_maps.items():
            # Resizing
            feat_map = F.interpolate(feat_map, size=self.featmap_size,
                                     mode=self.upsample,
                                     align_corners=self.align_corners)

            features.append(feat_map)

        # Concatenate to tensor
        features = torch.cat(features, dim=1)

        # Drop out feature maps
        features = features[:, self.feature_mask]

        return features


if __name__ == '__main__':
    feature_extractor = Extractor(
        backbone="resnet18",
        img_size=256,
        featmap_size=256
    )
    x = torch.randn(2, 3, 256, 256)
    y = feature_extractor(x)
    print(y.shape)

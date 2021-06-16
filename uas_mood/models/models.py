import torch


if __name__ == '__main__':
    model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2',
                            pretrained=True),
    print(model)

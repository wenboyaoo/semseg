import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=False))
        return torch.cat(out, 1)


class PSPNet(nn.Module):
    def __init__(self, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=32, use_ppm=True, backbone_name = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m", criterion=nn.CrossEntropyLoss(ignore_index=255)):
        super(PSPNet, self).__init__()
        assert 768 % len(bins) == 0
        assert classes > 1
        assert zoom_factor == 32
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        self.zoom_factor = zoom_factor
        self.backbone_name = backbone_name
        self.backbone = AutoModel.from_pretrained(self.backbone_name, output_hidden_states=True)
        self.hidden_layers = list(self.backbone.stages)
        for layer in self.hidden_layers:
            for param in layer.parameters():
                param.requires_grad = True

        fea_dim = 768
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )

    def forward(self, x, y=None):
        x_size = x.size()
        assert (x_size[2]) % 32 == 0 and (x_size[3]) % 32 == 0
        h = x_size[2]
        w = x_size[3]

        x = self.hidden_layers[0](x)
        x = self.hidden_layers[1](x)
        x_tmp = self.hidden_layers[2](x)
        x = self.hidden_layers[3](x_tmp)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=False)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x


if __name__ == '__main__':
    input = torch.rand(4, 3, 224, 224)
    model = PSPNet(bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=32, use_ppm=True)
    model.eval()
    print(model)
    output = model(input)
    print('PSPNet', output.size())

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

class FiLM(nn.Module):
    def __init__(self, token_length=768, hidden_dim=256, output_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, token):
        params = self.net(token)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        return gamma, beta


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, use_FiLM=True, token_length=768):
        super(PPM, self).__init__()
        self.features = []
        self.use_FiLM = use_FiLM
        self.token_length = token_length
        self.concat_channels = in_dim + len(bins) * reduction_dim
        if self.use_FiLM:
            self.film = FiLM(token_length=self.token_length, output_dim=2*self.concat_channels)
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x, token):
        x_size = x.size()
        out = [x]
        for f in self.features:
            pyramid_feat = f(x)
            upsampled_feat = F.interpolate(pyramid_feat, x_size[2:], mode='bilinear', align_corners=False)
            out.append(upsampled_feat)
        out = torch.cat(out, 1)
        if self.use_FiLM:
            gamma, beta = self.film(token)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            out = gamma * out + beta
    
        return out


class PSPNet(nn.Module):
    def __init__(self, bins=(1, 2, 3, 6), dropout=0.1, classes=2, use_fpn=True, use_ppm=True, use_FiLM=True, token_length=768, backbone_name = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m", criterion=nn.CrossEntropyLoss(ignore_index=255)):
        super(PSPNet, self).__init__()
        assert 512 % len(bins) == 0
        assert classes > 1
        self.use_ppm = use_ppm
        self.use_fpn = use_fpn
        self.use_FiLM = use_FiLM
        self.token_length = token_length
        self.criterion = criterion
        self.backbone_name = backbone_name

        self.backbone = AutoModel.from_pretrained(self.backbone_name, output_hidden_states=True).eval()
        if self.use_fpn:
            self.fpn = torchvision.ops.FeaturePyramidNetwork([96, 192, 384, 768], 1024)
            fea_dim = 1024
        else:
            fea_dim = 768
            bins = (1,2,3,4)
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, use_FiLM=self.use_FiLM, token_length=self.token_length)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )

    def forward(self, x, y=None):
        x_size = x.size()
        assert (x_size[2]) % 32 == 0 and (x_size[3]) % 32 == 0
        h = x_size[2]
        w = x_size[3]

        output = self.backbone(x)
        hiddeen_states = {i:s for i,s in enumerate(output.hidden_states) if i>0 }
        if self.use_fpn:
            x = self.fpn(hiddeen_states)[1]
        else:
            x = list(hiddeen_states.values())[-1]
        token = output.last_hidden_state[:, 0, :]
        if self.use_ppm:
            x = self.ppm(x,token)
        x = self.cls(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        if self.training:
            main_loss = self.criterion(x, y)
            return x.max(1)[1],main_loss
        else:
            return x

if __name__ == '__main__':
    input = torch.rand(4, 3, 224, 224)
    model = PSPNet(bins=(1, 2, 3, 6), dropout=0.1, classes=21, use_ppm=True, use_FiLM=False, use_fpn=False)
    model.eval()
    print(model)
    output = model(input)
    print('PSPNet', output.size())

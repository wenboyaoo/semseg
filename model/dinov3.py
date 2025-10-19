from torch import nn
from transformers import AutoModel

class DinoV3ConvNeXtTiny(nn.Module):
    def __init__(self, 
                model_name="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
                freeze_layers = None):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(self.model_name, output_hidden_states=True)

        self.layers = list(self.model.stages)
        self.freeze_layers = freeze_layers

        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = True
        if self.freeze_layers is not None:
            for i, layer in enumerate(self.layers):
                if i in self.freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

    def forward(self, x):
        return self.model(x)

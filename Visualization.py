import matplotlib.pyplot as plt
import visualtorch
from torch import nn
import torch


class RecognitionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3 * 64 * 64, 64 * 64),
            nn.Linear(64 * 64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 15),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = RecognitionNetwork()

tensor = (1, 3, 64, 64)

img = visualtorch.layered_view(model, input_shape=tensor, legend=True)
img1 = visualtorch.graph_view(model, input_shape=tensor,node_size=20,node_spacing=0)


plt.tight_layout()
plt.figure(figsize=(1.8,1),dpi=1000)
plt.axis("off")
plt.imshow(img)
plt.savefig("img",bbox_inches="tight", pad_inches=0)
plt.imshow(img1)
plt.savefig("img2", bbox_inches="tight", pad_inches=0)

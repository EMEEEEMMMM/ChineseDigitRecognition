import os
import torch
import pandas as pd
from torch import nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

torch.set_num_threads(2)
csv_file = (
    "/Change/it/to/your/path/ChineseDigitRecognition/data/chinese_mnist.csv"
)
project_dir = "/Change/it/to/your/path/ChineseDigitRecognition"
pic_dir = os.path.join(project_dir, "data/data")
character_to_label = {
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "百": 11,
    "千": 12,
    "万": 13,
    "亿": 14,
}


class ChineseDigitDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform):
        super().__init__()
        self.read_csv_file = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.read_csv_file["label"] = self.read_csv_file["character"].map(
            character_to_label
        )
        self.file_list = os.listdir(os.path.join(pic_dir, self.img_dir))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_name = self.file_list[index]
        image = self.transform(
            Image.open(os.path.join(pic_dir, self.img_dir, img_name)).convert("RGB")
        )
        name, pic_format = os.path.splitext(img_name)
        inquiry = name.split("_")
        result = self.read_csv_file[
            (self.read_csv_file["suite_id"] == int(inquiry[1]))
            & (self.read_csv_file["sample_id"] == int(inquiry[2]))
            & (self.read_csv_file["code"] == int(inquiry[3]))
        ]
        value_pic = result["label"].item()

        return image, torch.tensor(value_pic, dtype=torch.long)


transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ]
)

train_dataset = ChineseDigitDataset(
    csv_file=csv_file,
    img_dir="train_pic",
    transform=transform,
)

test_dataset = ChineseDigitDataset(
    csv_file=csv_file,
    img_dir="test_pic",
    transform=transform,
)


train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=4,shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, num_workers=4,shuffle=False)


class RecognitionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*64 * 64, 4096),
            nn.Linear(4096, 1024),
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

learning_rate = 1e-3
batch_size = 64
epochs = 100
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (image, value_pic) in enumerate(dataloader):
        pred = model(image)
        loss = loss_fn(pred, value_pic)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * batch_size + len(image)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")



def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for image, value_pic in dataloader:
            pred = model(image)
            test_loss += loss_fn(pred, value_pic).item()
            correct += (pred.argmax(1) == value_pic).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )

for t in range(epochs):
    print(f"Epoch {t+1} \n -------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

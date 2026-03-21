import os

# 屏蔽OpenMP重复初始化警告
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from my_loader import dataloader


parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_classes", type=int, default=3, help="the number of classes"
)  # 类别数
parser.add_argument(
    "--epochs", type=int, default=20, help="the number of training epoch"
)
parser.add_argument(
    "--batch_size", type=int, default=16, help="batch_size for training"
)
parser.add_argument("--lr", type=float, default=0.005, help="star learning rate")
parser.add_argument(
    "--data_path", type=str, default="D:/vscode/Iris-classification/Iris_data.txt"
)
parser.add_argument("--device", default="cpu", help="device id (i.e. 0 or 0,1 or cpu)")
opt = parser.parse_args()


class neuralnetwork(torch.nn.Module):
    def __init__(self, in_dim, hiddendim1, hiddendim2, out_dim):
        super().__init__()
        self.layer1 = torch.nn.Linear(in_dim, hiddendim1)
        self.layer2 = torch.nn.Linear(hiddendim1, hiddendim2)
        self.layer3 = torch.nn.Linear(hiddendim2, out_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


custom_dataset = dataloader(opt.data_path)
train_datasize = int(len(custom_dataset) * 0.7)
validation_datasize = int(len(custom_dataset) * 0.2)
test_datasize = len(custom_dataset) - train_datasize - validation_datasize
train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
    custom_dataset, [train_datasize, validation_datasize, test_datasize]
)

print(
    "Training set data size:",
    len(train_dataset),
    ",Validating set data size:",
    len(validation_dataset),
    ",Testing set data size:",
    len(test_dataset),
)

train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=None)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=opt.batch_size, shuffle=True
)
test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)


def infer_val(model, device, dataset=validation_dataloader):
    model.eval()
    acc_num = 0
    sap_num = 0
    with torch.no_grad():
        for datas in dataset:
            data, label = datas
            sap_num += len(label)
            output = model(data.to(device))
            pred = torch.max(output, dim=1)[1]
            acc_num += torch.eq(pred, label).sum().item()
            val_rate = acc_num / sap_num
            return val_rate


def infer_test(model, device, dataset=test_dataloader):
    model.eval()
    acc_num = 0
    sap_num = 0
    with torch.no_grad():
        for datas in dataset:
            data, label = datas
            sap_num += len(label)
            output = model(data.to(device))
            pred = torch.max(output, dim=1)[1]
            acc_num += torch.eq(pred, label).sum().item()
            test_rate = acc_num / sap_num
            return test_rate


def main(args):
    print(args)

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    model = neuralnetwork(4, 12, 6, 3)
    loss_function = torch.nn.CrossEntropyLoss()
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(pg, lr=args.lr)

    save_path = os.path.join(os.getcwd(), "weights")
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        acc_num = 0
        sap_num = 0
        train_bar = tqdm(train_dataloader, file=sys.stdout, ncols=200)

        for datas in train_bar:
            optimizer.zero_grad()  ##
            data, label = datas
            output = model(data.to(device))
            pred = torch.max(output, dim=1)[1]
            sap_num += len(label)
            acc_num += torch.eq(pred, label).sum().item()
            loss = loss_function(output, label.to(device))
            loss.backward()
            optimizer.step()
            acc_rate = acc_num / sap_num
            train_bar.desc = f"第{epoch}轮训练，正确率是{acc_rate}, 损失是{loss}。"

        d = infer_val(model, device)
        print(f"验证正确率是{d}")
        torch.save(model.state_dict(), os.path.join(save_path, "AlexNet.pth"))

    d = infer_test(model, device)
    print(f"测试正确率是{d}")


if __name__ == "__main__":
    main(opt)

import torch
import torch.nn as nn

# 定義兩層ANN
class NetANN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 定義第一層全連接層：輸入為 28*28 (MNIST 影像的展平大小)，輸出為 128 個神經元
        self.fc1 = nn.Linear(28 * 28, 128)

        # 定義 ReLU 激活函數，用於加入非線性特徵
        self.relu = nn.ReLU()
        # 定義第二層全連接層 (輸出層)：輸入為 128，輸出為 10 (對應 0~9 的數字分類)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 將輸入影像展平 (Flatten) 為 1D 向量
        x = x.view(-1, 28 * 28)
        
        # 依序通過第一層、激活函數以及輸出層
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

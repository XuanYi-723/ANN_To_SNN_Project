import torch
import torch.nn as nn
import snntorch as snn
# 定義脈衝神經網路 (SNN) 模型
class NetSNN(nn.Module):

    def __init__(self, ann_model):
        super().__init__()

        # 繼承 ANN 權重
        self.fc1 = ann_model.fc1
        self.fc2 = ann_model.fc2

        # Spike neuron
        self.lif1 = snn.Leaky(beta=0.9)
        self.lif2 = snn.Leaky(beta=0.9)

    def forward(self, x, num_steps=20):

        x = x.view(-1, 28 * 28)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_sum = torch.zeros(x.size(0), 10).to(x.device)

        total_spikes = 0

        for step in range(num_steps):

            # layer 1
            cur1 = self.fc1(x)

            spk1, mem1 = self.lif1(cur1, mem1)

            # layer 2
            cur2 = self.fc2(spk1)

            spk2, mem2 = self.lif2(cur2, mem2)

            # accumulate output spikes
            spk2_sum += spk2

            # spike count
            total_spikes += spk1.sum().item()
            total_spikes += spk2.sum().item()

        return spk2_sum, total_spikes

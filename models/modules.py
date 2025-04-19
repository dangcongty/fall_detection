import torch
import torch.nn as nn
import torch.utils


class SelfAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttentionLayer, self).__init__()
        self.query_conv = nn.Conv1d(input_dim, output_dim, kernel_size=1)
        self.key_conv = nn.Conv1d(input_dim, output_dim, kernel_size=1)
        self.value_conv = nn.Conv1d(input_dim, output_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # x shape: B x 34 x T
        Q = self.query_conv(x)  # B x 34 x T
        K = self.key_conv(x)    # B x 34 x T
        V = self.value_conv(x)  # B x 34 x T

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attention_weights = self.softmax(attention_scores)  # B x 34 x 34
        output = torch.matmul(attention_weights, V)  # B x 34 x T
        return output

class TemporalAttentionLayer(nn.Module):
    def __init__(self, in_chan):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_chan, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, T, V = x.size()
        x_feat = x.permute(0, 1, 3, 2).reshape(B, C*V, T)
        attn_wei = self.layer(x_feat).unsqueeze(-1) 
        x = x * attn_wei
        return x


class TemporalPart(nn.Module):
    def __init__(self, in_chan = 2):
        super().__init__()

        self.init_layer = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_chan),
            nn.LeakyReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_chan),
            nn.LeakyReLU(inplace=True),
        )
        self.attn1 = TemporalAttentionLayer(34)
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_chan),
            nn.LeakyReLU(inplace=True),
        )
        self.attn2 = TemporalAttentionLayer(34)

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_chan),
            nn.LeakyReLU(inplace=True),
        )
        self.attn3 = TemporalAttentionLayer(34)

        self.combine_layers = nn.Sequential(
            nn.ConvTranspose2d(in_chan, in_chan, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_chan),
            nn.LeakyReLU(inplace=True)
        )
        self.last_layers = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(in_chan),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_chan, in_chan, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(in_chan),
            nn.LeakyReLU(inplace=True),
        )
        
    def forward(self, x):
        T = x.shape[2]
        init_x = self.init_layer(x)
        x1 = self.layer1(init_x)
        x1_attn = self.attn1(x1)

        x2 = self.layer2(x1)
        x2_attn = self.attn2(x2)

        x3 = self.layer3(x2)
        x3_attn = self.attn3(x3)

        x_combine = torch.cat([x1_attn, x2_attn, x3_attn], dim=2)
        x_combine = self.combine_layers(x_combine)
        x_pool = torch.nn.functional.adaptive_avg_pool2d(x_combine, (T//2, 17))
        x_out = self.last_layers(x_pool)
        return x_out

class SpatialPart(nn.Module):
    def __init__(self):
        super().__init__()

        self.attn1 = SelfAttentionLayer(34, 34)
        self.attn2 = SelfAttentionLayer(34, 34)
        self.attn3 = SelfAttentionLayer(34, 34)

        self.last_layers = nn.Sequential(
            nn.Conv1d(34, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(16, 3, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        B, C, T, V = x.size()
        x = x.permute(0, 1, 3, 2).reshape(B, C*V, T)
        attn1 = self.attn1(x)
        attn2 = self.attn2(attn1)   
        attn3 = self.attn3(attn2)
        x_out = self.last_layers(attn3)
        return x_out
    

if __name__ == "__main__":
    model = STModel()
    x = torch.randn(2, 2, 50, 17)
    output = model(x)  # Output shape will be B x 34 x T
    print(output.shape)
    x = torch.randn(2, 2, 100, 17)
    output = model(x)  # Output shape will be B x 34 x T
    print(output.shape)
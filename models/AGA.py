import torch
import timm

from timm.models.vision_transformer import Block
from models.swin import SwinTransformer
from torch import nn
from einops import rearrange


class TABlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x


class SaveOutput:
    def __init__(self):
        self.outputs = []
    
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    
    def clear(self):
        self.outputs = []

import torch.nn.functional as F
class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConvolutionLayer, self).__init__()

        self.weight = nn.Parameter(torch.ones(in_channels, out_channels)*(1/out_channels))

        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.adjweight = nn.Parameter(torch.ones(784,784))

    def forward(self, x, adj_matrix):
        batch_size, in_channels, height, width = x.size()
        x = x.view(batch_size, in_channels, -1)
        adj_weight = adj_matrix * self.adjweight
        x = torch.bmm(x, adj_weight)
        x = x.permute(0,2,1)
        weight = self.weight.unsqueeze(0).repeat(batch_size,1,1)
        output = torch.bmm(x, weight)

        return output


class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphConvolutionalNetwork, self).__init__()
        self.gc1 = GraphConvolutionLayer(in_channels, hidden_channels)
        self.gc2 = GraphConvolutionLayer(hidden_channels, out_channels)

    def forward(self, x, adj_matrix):
        x = F.relu(self.gc1(x, adj_matrix))
        b,n,c = x.shape
        haha = torch.sqrt(torch.tensor(n))
        haha = int(haha)
        x = x.reshape(b,c,haha,haha)
        x = F.relu(self.gc2(x, adj_matrix))
        return x

channels = 768
hidden_channels = 384
output_channels = 768

def construct_adjacency_matrix(image_size):
    num_pixels = image_size * image_size
    adjacency_matrix = torch.zeros((num_pixels, num_pixels), dtype=torch.float)

    # Iterate over all pixels
    for i in range(num_pixels):
        # Get row and column indices of current pixel
        row = i // image_size
        col = i % image_size

        # Check if neighboring pixels are within the image boundaries
        if row > 0:
            adjacency_matrix[i, i - image_size] = 1.0  # Connect to pixel above
        if row < image_size - 1:
            adjacency_matrix[i, i + image_size] = 1.0  # Connect to pixel below
        if col > 0:
            adjacency_matrix[i, i - 1] = 1.0  # Connect to pixel on the left
        if col < image_size - 1:
            adjacency_matrix[i, i + 1] = 1.0  # Connect to pixel on the right

        # Connect to diagonal pixels
        if row > 0 and col > 0:
            adjacency_matrix[i, i - image_size - 1] = 1.0  # Connect to pixel on the top-left diagonal
        if row > 0 and col < image_size - 1:
            adjacency_matrix[i, i - image_size + 1] = 1.0  # Connect to pixel on the top-right diagonal
        if row < image_size - 1 and col > 0:
            adjacency_matrix[i, i + image_size - 1] = 1.0  # Connect to pixel on the bottom-left diagonal
        if row < image_size - 1 and col < image_size - 1:
            adjacency_matrix[i, i + image_size + 1] = 1.0  # Connect to pixel on the bottom-right diagonal

        # Connect to self
        # adjacency_matrix[i, i] = 1.0

    return adjacency_matrix

def continuous2discrete(score, d_min, d_max, n_c):
    score = torch.round((score - d_min) / (d_max - d_min) * (n_c - 1))
    return score

def discrete2continuous(score, d_min, d_max, n_c):
    score = score / (n_c - 1) * (d_max - d_min) + d_min
    return score

class AGAIQA(nn.Module):
    def __init__(self, embed_dim=72, num_outputs=1, patch_size=8, drop=0.2,
                    depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                    img_size=224, num_tab=2, scale=0.8, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        
        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        self.tablock1 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock1.append(tab)

        self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.tablock2 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock2.append(tab)

        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.swintransformer2 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim // 2,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )
        self.tablock3 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock3.append(tab)

        self.conv3 = nn.Conv2d(embed_dim // 2, embed_dim // 4, 1, 1, 0)
        self.swintransformer3 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim // 4,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )
        
        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.Sigmoid()
        )

        self.fc_score_or = nn.Sequential(
            nn.Linear(embed_dim // 4, embed_dim // 8),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 8, 64),
            nn.ReLU()
        )
        self.fc_weight_or = nn.Sequential(
            nn.Linear(embed_dim // 4, embed_dim // 8),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 8, 64),
            nn.Sigmoid()
        )


        self.gcn = GraphConvolutionalNetwork(channels, hidden_channels, output_channels).cuda()
        self.gcn2 = GraphConvolutionalNetwork(384, hidden_channels, 384).cuda()
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(0.2)
        self.w2.data.fill_(0.2)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3.data.fill_(0.9)
        self.w4.data.fill_(0.1)

    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x

    def soft_ordinal_regression(self, pred_prob, d_min, d_max, n_c):
        pred_prob_sum = torch.sum(pred_prob, 1, keepdim=True)
        Intergral = torch.floor(pred_prob_sum)
        Fraction = pred_prob_sum - Intergral
        score_low = (discrete2continuous(Intergral, d_min, d_max, n_c) +
                     discrete2continuous(Intergral + 1, d_min, d_max, n_c)) / 2
        score_high = (discrete2continuous(Intergral + 1, d_min, d_max, n_c) +
                      discrete2continuous(Intergral + 2, d_min, d_max, n_c)) / 2
        pred_score = score_low * (1 - Fraction) + score_high * Fraction
        return pred_score

    def decode_ord(self, y):
        batch_size, prob = y.shape
        y = torch.reshape(y, (batch_size, prob // 2, 2, 1, 1))
        denominator = torch.sum(y, 2) + 0.0000005
        pred_score = torch.div(y[:, :, 1, :, :], denominator)
        return pred_score


    def inference(self, y):

        inferenceFunc = self.soft_ordinal_regression
        pred_score = inferenceFunc(y, 0.0, 1.0, 32)
        return pred_score

    def forward(self, x):
        _x = self.vit(x)
        x = self.extract_feature(self.save_output)
        self.save_output.outputs.clear()

        # stage 1
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        # x_or = x
        for tab in self.tablock1:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        x = self.swintransformer1(x)

        b,c,h,w = x.shape
        adj_matrix = construct_adjacency_matrix(28)
        adj_matrix.unsqueeze(0)
        adj_matrix = adj_matrix.repeat(b,1,1).cuda()
        x1 = self.gcn(x,adj_matrix)
        x1 = x1.reshape(b,768,28,28)
        x = x * 0.8 + x1 * self.w1

        # stage2
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        # x_or = x
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        x = self.swintransformer2(x)

        b,c,h,w = x.shape
        adj_matrix = construct_adjacency_matrix(28)
        adj_matrix.unsqueeze(0)
        adj_matrix = adj_matrix.repeat(b,1,1).cuda()
        x2 = self.gcn2(x,adj_matrix)
        x2 = x2.reshape(b,384,28,28)
        x = x * 0.9 + x2 * self.w2

        print(self.w1, self.w2,'the weight of w1 and w2')

        # deep ordinal
        x_or = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock3:
            x_or = tab(x_or)
        x_or = rearrange(x_or, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x_or = self.conv3(x_or)
        x_or = self.swintransformer3(x_or)

        x_or = rearrange(x_or, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)

        f_or = self.fc_score_or(x_or)
        w_or = self.fc_weight_or(x_or)
        # print(f_or.shape)
        pred = (f_or * w_or).sum(dim=1) / w_or.sum(dim=1)
        pred = self.decode_ord(pred)
        score1 = self.inference(pred)



        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)

        score = self.w3 * score + self.w4 * score1.squeeze()
        return score, pred

import torch
from torch import nn


class BinaryTreeLstmCell(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.h_dim = hidden_dim
        self.linear = nn.Linear(in_features=2 * self.h_dim, out_features=5 * self.h_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        # add some positive bias for the forget gates [b_g, b_i, b_f, b_f, b_o] = [0, 0, 1, 1, 0]
        nn.init.constant_(self.linear.bias, val=0)
        nn.init.constant_(self.linear.bias[2 * self.h_dim:4 * self.h_dim], val=1)

    def forward(self, h_l, c_l, h_r, c_r):
        # h_l, c_l [B, L-1, hidden_dim]
        # h_r, c_r [B, L-1, hidden_dim]

        h_lr = torch.cat([h_l, h_r], dim=-1)
        # h_lr [B, L-1, 2 * hidden_dim]

        g, i, f_l, f_r, o = self.linear(h_lr).chunk(chunks=5, dim=-1)
        g, i, f_l, f_r, o = g.tanh_(), i.sigmoid_(), f_l.sigmoid_(), f_r.sigmoid_(), o.sigmoid_()
        # g, i, f_l, f_r, o [B, L-1, hidden_dim]

        c = i * g + f_l * c_l + f_r * c_r

        h = o * c.tanh_()
        # h, c [B, L-1, hidden_dim]
        return h, c
import torch
from torch import nn

from model.BinaryTreeLstmCell import BinaryTreeLstmCell


class LstmRnn(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.i_dim = input_dim
        self.h_dim = hidden_dim
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.h0 = nn.Parameter(torch.empty(size=(1, hidden_dim), dtype=torch.float32))
        self.c0 = nn.Parameter(torch.empty(size=(1, hidden_dim), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.h0, val=0)
        nn.init.constant_(self.c0, val=0)

        nn.init.xavier_uniform_(self.lstm.weight_ih)
        nn.init.orthogonal_(self.lstm.weight_hh)
        nn.init.constant_(self.lstm.bias_ih, val=0)
        nn.init.constant_(self.lstm.bias_hh, val=0)

    def forward(self, x, mask, backward=False):
        # x: [B, L, word_dim]
        # mask: [B, L]
        L = x.shape[1]  # length of the sequence
        prev_h = self.h0.expand(x.shape[0], -1)
        prev_c = self.c0.expand(x.shape[0], -1)
        # prev_h [B, hidden_dim]
        # prev_c [B, hidden_dim]
        h = []
        for idx in range(L):
            idx = L - 1 - idx if backward else idx
            mask_idx = mask[:, idx, None]
            # mask_idx: the column at idx but with size[B, 1] not [B]
            h_idx, c_idx = self.lstm(x[:, idx], (prev_h, prev_c))
            # if location idx is not masked, prev_h will be updated; otherwise, prev_h stays the same
            prev_h = h_idx * mask_idx + prev_h * (1. - mask_idx)
            prev_c = c_idx * mask_idx + prev_c * (1. - mask_idx)
            # prev_h [B, hidden_dim]
            h.append(prev_h)
        # return [B, L, hidden_dim]
        return torch.stack(h[::-1] if backward else h, dim=1) # stack all the h together


class BinaryTreeBasedModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob, trans_hidden_dim):
        super().__init__()
        self.trans_lstm = LstmRnn(input_dim, trans_hidden_dim)
        self.trans_linear = nn.Linear(in_features=trans_hidden_dim, out_features=2 * hidden_dim)
        self.tree_lstm_cell = BinaryTreeLstmCell(hidden_dim, dropout_prob)
        BinaryTreeBasedModule.reset_parameters(self)

    def reset_parameters(self):
        nn.init.orthogonal_(self.trans_linear.weight)
        nn.init.constant_(self.trans_linear.bias, val=0)
        self.tree_lstm_cell.reset_parameters()
        self.trans_lstm.reset_parameters()

    def forward(self, *inputs):
        raise NotImplementedError


    def _transform_leafs(self, x, mask):
        # TODO(siyu) adding more leaf transformation methods
        # x: [B, L, input_dim]
        # mask: [B, L]
        # return two [B, L, hidden_dim] tensors --> h, c
        x = self.trans_lstm(x, mask)
        return self.trans_linear(x).tanh().chunk(chunks=2, dim=-1)

    @staticmethod
    def _merge(actions, h_l, c_l, h_r, c_r, h_p, c_p, mask):
        """
        This method merges left and right TreeLSTM states. It reuses already precomputed states for the parent node,
        but still, has to apply correct masking.
        """
        cumsum = torch.cumsum(actions, dim=-1)      # [B, L-k, 1] [0, 0, 1, 1, 1]
        mask_l = (1.0 - cumsum)[..., None]          # [B, L-k, 1] [1, 1, 0, 0, 0]   # adding one dimension so that if can * elementwise
        mask_r = (cumsum - actions)[..., None]      # [B, L-k, 1] [0, 0 ,0, 1, 1]
        mask = mask[..., None]                      # [B, L-k, 1] [1, 1, 1, 1, 1]
        actions = actions[..., None]                # [B, L-k, 1] [0, 0, 1, 0, 0]

        # If the row of mask matrix is zero ignore everything calculated so far and copy the corresponding left hidden
        # states from the previous layer (the assumption here is that one adds padding tokens to the right side and
        # action that uses padding token can't be sampled if the row of a mask is a nonzero vector).
        # Eventually, you will end up with the leftmost state on the top that contains a correct required value.

        # TODO(siyu): why adding left for masked positions, should have no effect
        h_p = (mask_l * h_l + actions * h_p + mask_r * h_r) * mask + h_l * (1. - mask)
        c_p = (mask_l * c_l + actions * c_p + mask_r * c_r) * mask + c_l * (1. - mask)
        return h_p, c_p

import torch
from torch import nn

from model.BinaryTreeLstmCell import BinaryTreeLstmCell


class BinaryTreeBasedModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=2 * hidden_dim)
        self.tree_lstm_cell = BinaryTreeLstmCell(hidden_dim)
        BinaryTreeBasedModule.reset_parameters(self)

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, val=0)
        self.tree_lstm_cell.reset_parameters()

    def forward(self, *inputs):
        raise NotImplementedError


    def _transform_leafs(self, x, mask):
        # TODO(siyu) adding more leaf transformation methods
        # x: [B, L, input_dim]
        # mask: [B, L]
        # return two [B, L, hidden_dim] tensors --> h, c
        return self.linear(x).tanh().chunk(chunks=2, dim=-1)

    # TODO(siyu) purpose of mask
    @staticmethod
    def _merge(actions, h_l, c_l, h_r, c_r, h_p, c_p, mask):
        """
        This method merges left and right TreeLSTM states. It reuses already precomputed states for the parent node,
        but still, has to apply correct masking.
        """
        cumsum = torch.cumsum(actions, dim=-1)      # [B, L-k, 1] [0, 0, 1, 1, 1]
        mask_l = (1.0 - cumsum)[..., None]          # [B, L-k, 1] [1, 1, 0, 0, 0]
        mask_r = (cumsum - actions)[..., None]      # [B, L-k, 1] [0, 0 ,0, 1, 1]
        mask = mask[..., None]                      # [B, L-k, 1] [1, 1, 1, 1, 1]
        actions = actions[..., None]                # [B, L-k, 1] [0, 0, 1, 0, 0]

        # If the row of mask matrix is zero ignore everything calculated so far and copy the corresponding left hidden
        # states from the previous layer (the assumption here is that one adds padding tokens to the right side and
        # action that uses padding token can't be sampled if the row of a mask is a nonzero vector).
        # Eventually, you will end up with the leftmost state on the top that contains a correct required value.

        h_p = (mask_l * h_l + actions * h_p + mask_r * h_r) * mask + h_l * (1. - mask)
        c_p = (mask_l * c_l + actions * c_p + mask_r * c_r) * mask + c_l * (1. - mask)
        return h_p, c_p
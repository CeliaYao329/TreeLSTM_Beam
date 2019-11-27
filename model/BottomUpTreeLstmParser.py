import math
import torch
from torch import nn
from model.utils import Categorical
from model.BinaryTreeBasedModule import BinaryTreeBasedModule


class BottomUpTreeLstmParser(BinaryTreeBasedModule):
    def __init__(self, input_dim, hidden_dim, dropout_prob, trans_hidden_dim):
        super().__init__(input_dim, hidden_dim, dropout_prob, trans_hidden_dim)
        self.q = nn.Parameter(torch.empty(size=(hidden_dim,), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.q, mean=0, std=0.01)

    def forward(self, x, mask, eval_actions=None):
        probs = []
        gumbel_noise = []
        actions = []
        entropy = []
        normalized_entropy = []
        log_prob = []

        # h [B, L, hidden_dim]
        # c [B, L, hidden_dim]
        h, c = self._transform_leafs(x, mask)
        for i in range(1, x.shape[1]):  # 1, 2, ... , L-1
            ev_actions_i = None if eval_actions is None else eval_actions[i - 1]
            cat_distribution, gumbel_noise_i, actions_i, h, c = self._make_step(h, c, mask[:, i:], ev_actions_i)
            probs.append(cat_distribution.cat_distr.probs)
            gumbel_noise.append(gumbel_noise_i)
            actions.append(actions_i)
            entropy.append(cat_distribution.entropy())
            normalized_entropy.append(cat_distribution.normalized_entropy())
            log_prob.append(cat_distribution.log_prob(actions_i))
        log_prob = sum(log_prob)    # [B]
        entropy = sum(entropy)
        # normalize by the number of layers - 1.
        # -1 because the last layer contains only one possible action and the entropy is zero anyway.

        # TODO: normalized entropy are all 1
        normalized_entropy = sum(normalized_entropy) / (torch.sum(mask[:, 2:], dim=-1) + 1e-17)
        return probs, entropy, normalized_entropy, actions, log_prob

    def _make_step(self, h, c, mask, ev_actions):
        # mask [B, L-k]
        h_l, c_l = h[:, :-1, :], c[:, :-1, :]
        h_r, c_r = h[:, 1:, :], c[:, 1:, :]
        h_p, c_p = self.tree_lstm_cell(h_l, c_l, h_r, c_r)
        # score: (B x L-k x hidden_dim, hidden_dim) -> [B x L-k]
        score = torch.matmul(h_p, self.q)
        cat_distr = Categorical(score, mask)
        if ev_actions is None:
            actions, gumbel_noise = self._sample_action(cat_distr, mask)
        else:
            actions = ev_actions
            gumbel_noise = None
        h_p, c_p = BinaryTreeBasedModule._merge(actions, h_l, c_l, h_r, c_r, h_p, c_p, mask)
        return cat_distr, gumbel_noise, actions, h_p, c_p

    def _sample_action(self, cat_distri, mask):

        if self.training:
            actions, gumbel_noise = cat_distri.rsample()
        else:
            actions = torch.zeros_like(cat_distri.cat_distr.probs)
            actions.scatter(-1, torch.argmax(cat_distri.cat_distr.probs, dim=-1, keepdim=True), 1.0)
            gumbel_noise = None
        return actions, gumbel_noise

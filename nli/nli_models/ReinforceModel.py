from itertools import chain
import torch
from torch import nn
from torch.nn import functional as F

from model.BottomUpTreeLstmParser import BottomUpTreeLstmParser
from model.BinaryTreeLstmRnn import BinaryTreeLstmRnn


class ReinforceModel(nn.Module):
    def __init__(self, vocab_size, word_dim, hidden_dim, mlp_hidden_dim, label_dim, dropout_prob, trans_hidden_dim, use_batchnorm=False, baseline_type=None):
        super().__init__()
        self.parser_embedding = nn.Embedding(vocab_size, word_dim)
        self.parser = BottomUpTreeLstmParser(word_dim, hidden_dim, dropout_prob, trans_hidden_dim)
        self.tree_embedding = nn.Embedding(vocab_size, word_dim)
        self.tree_lstm_rnn = BinaryTreeLstmRnn(word_dim, hidden_dim, dropout_prob, trans_hidden_dim)

        self.dropout = nn.Dropout(dropout_prob)
        self.linear1 = nn.Linear(in_features=4 * hidden_dim, out_features=mlp_hidden_dim)
        self.linear2 = nn.Linear(in_features=mlp_hidden_dim, out_features=label_dim)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn1 = nn.BatchNorm1d(num_features=4 * hidden_dim)
            self.bn2 = nn.BatchNorm1d(num_features=mlp_hidden_dim)

        self.baseline = baseline_type
        if self.baseline is not None and self.baseline not in ("ema", "self_critical"):
            raise ValueError(f"There is no baseline type as {baseline_type}.")
        if self.baseline == "ema":
            self.ema_mean = 0.48
            self.ema_alpha = 0.9

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.parser_embedding.weight, 0.0, 0.01)
        nn.init.normal_(self.tree_embedding.weight, 0.0, 0.01)
        self.parser.reset_parameters()
        self.tree_lstm_rnn.reset_parameters()

        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, val=0)
        nn.init.uniform_(self.linear2.weight, -0.005, 0.005)
        nn.init.constant_(self.linear2.bias, val=0)
        if self.use_batchnorm:
            self.bn1.reset_parameters()
            self.bn2.reset_parameters()

    def forward(self, premises, p_mask, hypotheses, h_mask, labels):
        entropy, normalized_entropy, actions, actions_log_prob, logits, rewards = self._forward(
            premises, p_mask, hypotheses, h_mask, labels)
        ce_loss = rewards.mean()
        if self.training:
            baseline = self._get_baseline(rewards, premises, p_mask, hypotheses, h_mask, labels)
            rewards = rewards - baseline
        pred_labels = logits.argmax(dim=1)
        return pred_labels, ce_loss, rewards.detach(), actions, actions_log_prob, entropy, normalized_entropy

    def _forward(self, premises, p_mask, hypotheses, h_mask, labels):
        p_parser_embed = self.dropout(self.parser_embedding(premises))
        h_parser_embed = self.dropout(self.parser_embedding(hypotheses))
        _, p_entropy, p_normalized_entropy, p_actions, p_actions_log_prob = self.parser(p_parser_embed, p_mask)
        _, h_entropy, h_normalized_entropy, h_actions, h_actions_log_prob = self.parser(h_parser_embed, h_mask)

        p_tree_embed = self.dropout(self.tree_embedding(premises))
        h_tree_embed = self.dropout(self.tree_embedding(hypotheses))
        p_final_h = self.tree_lstm_rnn(p_tree_embed, p_actions, p_mask)
        h_final_h = self.tree_lstm_rnn(h_tree_embed, h_actions, h_mask)

        h = torch.cat((p_final_h, h_final_h, (p_final_h - h_final_h).abs(), p_final_h * h_final_h), dim=1)
        if self.use_batchnorm:
            h = self.bn1(h)
        h = self.linear1(self.dropout(h))
        h = F.relu(h)
        if self.use_batchnorm:
            h = self.bn2(h)
        h = self.dropout(h)
        logits = self.linear2(h)
        rewards = self.criterion(input=logits, target=labels)

        actions = {"p_actions": p_actions, "h_actions": h_actions}
        actions_log_prob = p_actions_log_prob + h_actions_log_prob
        entropy = (p_entropy + h_entropy) / 2.0
        normalized_entropy = (p_normalized_entropy + h_normalized_entropy) / 2.0

        return entropy, normalized_entropy, actions, actions_log_prob, logits, rewards

    def get_policy_parameters(self):
        if self.parser_embedding.weight.requires_grad:
            return list(chain(self.parser_embedding.parameters(), self.parser.parameters()))
        return list(self.parser.parameters())

    def get_environment_parameters(self):
        if self.tree_embedding.weight.requires_grad:
            return list(
                chain(self.tree_embedding.parameters(), self.tree_lstm_rnn.parameters(), self.linear1.parameters(),
                      self.linear2.parameters()))
        return list(chain(self.tree_lstm_rnn.parameters(), self.linear1.parameters(), self.linear2.parameters()))

    def _get_baseline(self, rewards, premises, p_mask, hypotheses, h_mask, labels):
        if self.baseline is None:
            return 0.0
        # TODO: figure out baseline
        with torch.no_grad():
            if self.baseline == "ema":
                cur_mean = self.baseline_mean
                self.baseline_mean = self.baseline_mean * self.baseline_alpha + rewards.mean() * (1.0 - self.baseline_alpha)
                return cur_mean
            elif self.baseline == "self_critical":
                self.eval()
                self.dropout.train()
                rewards = self._forward(premises, p_mask, hypotheses, h_mask, labels)[-1]
                self.train()
                return rewards

from model.BinaryTreeBasedModule import BinaryTreeBasedModule


class BinaryTreeLstmRnn(BinaryTreeBasedModule):
    def __init__(self, input_dim, hidden_dim, dropout_prob, trans_hidden_dim):
        super().__init__(input_dim, hidden_dim, dropout_prob, trans_hidden_dim)

    def forward(self, x, parse_tree, mask):
        h, c = self._transform_leafs(x, mask)
        for i in range(x.shape[1] - 1):
            h_l, c_l = h[:, :-1], c[:, :-1]
            h_r, c_r = h[:, 1:], c[:, 1:]
            h_p, c_p = self.tree_lstm_cell(h_l, c_l, h_r, c_r)
            h, c = self._merge(parse_tree[i], h_l, c_l, h_r, c_r, h_p, c_p, mask[:, i + 1:])
        return h.squeeze(dim=1)
from nli.nli_models.ReinforceModel import ReinforceModel

class PpoModel(ReinforceModel):
    def evaluataion_actions(self, premises, p_mask, p_actions, hypotheses, h_mask, h_actions):
        p_parser_embed = self.parser_embedding(premises)
        h_parser_embed = self.parser_embedding(premises)
        p_normalized_entropy, _, p_actions_log_prob = self.parser(p_parser_embed, p_mask, eval_actions=p_actions)[2:]
        h_normalized_entropy, _, h_actions_log_prob = self.parser(h_parser_embed, h_mask, eval_actions=h_actions)[2:]
        actions_log_prob = p_actions_log_prob + h_actions_log_prob
        normalized_entropy = (p_normalized_entropy + h_normalized_entropy) / 2.0

        return normalized_entropy, actions_log_prob
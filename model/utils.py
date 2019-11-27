import torch
from torch.distributions import utils as distr_utils
from torch.nn import functional as F
from torch.distributions.categorical import Categorical as TorchCategorical


class AverageMeter:
    def __init__(self):
        self.value = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


class Categorical:
    def __init__(self, scores, mask):
        # TODO(siyu): mask and n and log_n
        self.mask = mask
        self.n = self.mask.sum(dim=-1)
        self.log_n = (self.n + 1e-17).log()
        self.cat_distr = TorchCategorical(Categorical.masked_softmax(scores, self.mask))

    @staticmethod
    def masked_softmax(logits, mask):
        probs = F.softmax(logits, dim=-1) * mask
        probs = probs + (mask.sum(dim=-1, keepdim=True) == 0.).to(dtype=torch.float32)  # in case the are all masked for some data points
        z = probs.sum(dim=-1, keepdim=True)
        return probs / z

    def rsample(self):
        with torch.no_grad():
            uniforms = torch.empty_like(self.cat_distr.probs).uniform_()
            uniforms = distr_utils.clamp_probs(uniforms)
            # uniforms = distr_utils.clamp_probs(uniforms)
            gumbel_noise = -(-uniforms.log()).log()
            scores = (self.cat_distr.logits + gumbel_noise)
            scores = Categorical.masked_softmax(scores, self.mask)
            sample = torch.zeros_like(scores)
            sample.scatter_(-1, scores.argmax(dim=-1, keepdim=True), 1.0) # --> action index with 1 others being 0
            return sample, gumbel_noise

    def entropy(self):
        entropy = - torch.sum(self.cat_distr.logits * self.cat_distr.probs * self.mask, dim=-1)
        more_than_category = (self.n != 1.0).to(dtype = torch.float32)
        # to make sure that the entropy is precisely zero when there is only one category
        return entropy * more_than_category

    def normalized_entropy(self):
        return self.entropy() / (self.log_n + 1e-17)

    def log_prob(self, value):
        max_value, max_idxs = value.max(dim=-1)
        return self.cat_distr.log_prob(max_idxs) * (self.n != 0.).to(dtype=torch.float32)   # when they are all masked, log_prob should not be added more.

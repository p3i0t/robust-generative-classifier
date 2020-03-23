import torch
import torch.nn.functional as F
import torch.nn as nn
import math

import resnet


from losses.dim_losses import donsker_varadhan_loss, infonce_loss, fenchel_dual_loss
from mi_networks import MI1x1ConvNet


def cal_parameters(model):
    cnt = 0
    for para in model.parameters():
        cnt += para.numel()
    return cnt


class ClassConditionalGaussianMixture(nn.Module):
    def __init__(self, n_classes, embed_size):
        super().__init__()
        self.n_classes = n_classes
        self.embed_size = embed_size
        self.class_embed = nn.Embedding(n_classes, embed_size * 2)
        #nn.init.xavier_uniform_(self.class_embed.weight)

    def log_lik(self, x, mean, log_sigma):
        tmp = math.log(2 * math.pi) + 2 * log_sigma + (x - mean).pow(2) * torch.exp(-2 * log_sigma)
        ll = -0.5 * tmp
        return ll

    def forward(self, x):
        # create all class labels for each sample x
        y_full = torch.arange(self.n_classes).unsqueeze(dim=0).repeat(x.size(0), 1).view(-1).to(x.device)

        # repeat each sample for n_classes times
        x = x.unsqueeze(dim=1).repeat(1, self.n_classes, 1).view(x.size(0) * self.n_classes, -1)

        mean, log_sigma = torch.split(self.class_embed(y_full), split_size_or_sections=self.embed_size, dim=-1)

        # evaluate log-likelihoods for each possible (x, y) pairs
        ll = self.log_lik(x, mean, log_sigma).sum(dim=-1).view(-1, self.n_classes)
        return ll


def compute_dim_loss(l_enc, m_enc, measure, mode):
    '''Computes DIM loss.
    Args:
        l_enc: Local feature map encoding.
        m_enc: Multiple globals feature map encoding.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''

    if mode == 'fd':
        loss = fenchel_dual_loss(l_enc, m_enc, measure=measure)
    elif mode == 'nce':
        loss = infonce_loss(l_enc, m_enc)
    elif mode == 'dv':
        loss = donsker_varadhan_loss(l_enc, m_enc)
    else:
        raise NotImplementedError(mode)

    return loss


class SDIM(torch.nn.Module):
    def __init__(self, rep_size=64, n_classes=10, mi_units=128, encoder_name='resnet10', image_channel=1, margin=5,
                 alpha=0.33, beta=0.33, gamma=0.33):
        super().__init__()
        self.rep_size = rep_size
        self.n_classes = n_classes
        # self.input_shape = input_shape
        self.mi_units = mi_units
        self.encoder_name = encoder_name
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # build encoder
        n = int(encoder_name.strip('resnet'))
        self.encoder = resnet.build_resnet_32x32(n, fc_size=rep_size, image_channel=image_channel)  # output a representation
        print('==> # encoder parameters {}'.format(cal_parameters(self.encoder)))

        self.task_idx = (2, -1)

        local_size = 64

        # 1x1 conv performed on only channel dimension
        self.local_MInet = MI1x1ConvNet(local_size, self.mi_units)
        self.global_MInet = MI1x1ConvNet(self.rep_size, self.mi_units)

        self.class_conditional = ClassConditionalGaussianMixture(self.n_classes, self.rep_size)
        #self.class_conditional = ClassConditionalMAF(self.n_classes, self.rep_size)

    def _T(self, out_list):
        L, G = [out_list[i] for i in self.task_idx]

        # All globals are reshaped as 1x1 feature maps.
        global_size = G.size()[1:]
        if len(global_size) == 1:
            G = G[:, :, None, None]

        L = self.local_MInet(L)
        G = self.global_MInet(G)

        N, local_units = L.size()[:2]
        L = L.view(N, local_units, -1)
        G = G.view(N, local_units, -1)
        return L, G

    def eval_losses(self, x, y, measure='JSD', mode='fd'):
        out_list = self.encoder(x, return_full_list=True)
        rep = out_list[-1]
        L, G = self._T(out_list)

        # compute mutual infomation loss
        mi_loss = compute_dim_loss(L, G, measure, mode)

        # evaluate log-likelihoods as logits
        ll = self.class_conditional(rep) / self.rep_size

        pos_mask = torch.zeros(x.size(0), self.n_classes).to(x.device).scatter(1, y.unsqueeze(dim=1), 1.)

        # compute nll loss
        nll_loss = -(ll * pos_mask).sum(dim=1).mean()

        pos_ll = torch.masked_select(ll, pos_mask.byte())
        assert pos_ll.size(0) == x.size(0)
        gap_ll = pos_ll.unsqueeze(dim=1) - ll

        # log-likelihood margin loss
        ll_margin = F.relu(self.margin - gap_ll).pow(2).mean()

        # total loss
        loss = self.alpha * mi_loss + self.beta * nll_loss + self.gamma * ll_margin
        return loss, mi_loss, nll_loss, ll_margin

    def forward(self, x, log_softmax=False):
        rep = self.encoder(x, return_full_list=True)[-1]
        log_lik = self.class_conditional(rep)
        if log_softmax:
            return F.log_softmax(log_lik, dim=-1)
        return log_lik


if __name__ == '__main__':
    # rep_size = 64
    # mi_size = 512
    model = SDIM()
    print('==> # parameters {}'.format(cal_parameters(model)))
    x = torch.randn(5, 3, 32, 32)
    y = (torch.rand(5)*10).long()
    loss, mi_loss, nll_loss, margin = model(x, y)
    print('loss ', loss)

import torch
import torch.nn.functional as F
from torch.optim import Adam


def cw(model, images, labels, targeted=False, c=10., max_iter=1000, learning_rate=0.001, confidence=0.):
    def loss_fn(x, y):
        logits = model(x)

        one_hots = torch.eye(len(logits[0]))[y].to(x.device)

        real = (logits * one_hots).sum(dim=1)   # (b, 1)
        other = ((1 - one_hots) * logits - one_hots * 1e4).max(dim=1)[0]   # select the second largest logits

        if targeted:
            # if targeted, optimize for making the second largest logit larger than target logit
            return torch.clamp(other - real + confidence, min=0.)
        else:
            return torch.clamp(real - other + confidence, min=0.) #+ 1000 * torch.clamp(430. - other + confidence, min=0)
            #return torch.clamp(430. - other + confidence, min=0)

    w = torch.zeros_like(images, requires_grad=True).to(images.device)

    optimizer = Adam([w], lr=learning_rate)

    prev = 1e10

    for step in range(max_iter):
        new_img = (torch.tanh(w) + 1) / 2
        l2_dist = F.mse_loss(new_img, images, reduction='sum')
        fn_loss = loss_fn(new_img, labels).sum()

        cost = fn_loss + l2_dist

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter // 10) == 0:
            if cost > prev:
                print('Attack Stopped due to CONVERGENCE....')
                return new_img
            prev = cost

            print('step: {}, fn_loss: {:.3f}, l2_dist: {:.3f}'.format(step, fn_loss.item(), l2_dist.item()))
    adv_examples = (torch.tanh(w) + 1) / 2
    noises = adv_examples - images
    adv_logits = model(adv_examples)

    return adv_examples, noises, adv_logits

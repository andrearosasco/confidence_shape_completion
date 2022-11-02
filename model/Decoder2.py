import torch
from torch.nn import BCEWithLogitsLoss
from configs import Config
from utils.sgdiff import DifferentiableSGD


class Decoder:
    def __init__(self, sdf):

        self.sdf = sdf

    def __call__(self, fast_weights):
        old = self.sdf.training
        self.sdf.eval()

        batch_size = fast_weights[0][0].shape[0]
        refined_pred = torch.tensor(torch.randn(batch_size, 10000, 3).cpu().detach().numpy() * 0.1, device=Config.General.device,
                                    requires_grad=True)
        refined_pred.retain_grad()
        fast_weights = [[t.requires_grad_(True) for t in l] for l in fast_weights]
        for layer in fast_weights:
            for w in layer:
                w.retain_grad()

        loss_function = BCEWithLogitsLoss(reduction='mean')

        c1, c2, c3, c4 = 1, 0, 0, 0 #1, 0, 0  1, 1e3, 0 # 0, 1e4, 5e2
        # refined_pred.detach().clone()
        for step in range(20):
            optim = DifferentiableSGD([refined_pred], lr=0.001)
            results = self.sdf(refined_pred, fast_weights)

            gt = torch.ones_like(results[..., 0], dtype=torch.float32)
            gt[:, :] = 1
            loss1 = c1 * loss_function(results[..., 0], gt)

            loss_value = loss1

            self.sdf.zero_grad()
            loss_value.backward(retain_graph=True)
            refined_pred = optim.step()[0]
            refined_pred.retain_grad()

        return refined_pred


import torch
from torch.autograd import Function
import torch.nn as nn
import scipy



class netLoss():

    def __init__(self, args, masked_kspace=True):
        self.args = args

        mask_path = args.mask_path
        mat = scipy.io.loadmat(mask_path)

        self.masked_kspace = masked_kspace
        self.mask = torch.tensor(mat['mask_2'], device=self.args.device)
        self.maskNot = self.mask == 0


        self.ImL2_weights = args.LossWeights[0]
        self.ImL1_weights = args.LossWeights[1]
        self.KspaceL2_weights = args.LossWeights[2]

        self.ImL2Loss = nn.MSELoss()
        self.ImL1Loss = nn.SmoothL1Loss()
        if self.masked_kspace:
            self.KspaceL2Loss = nn.MSELoss(reduction='sum')
        else:
            self.KspaceL2Loss = nn.MSELoss()

    def calc(self, pred_Im, pred_K, tar_Im, tar_K):
        ImL2 = self.ImL2Loss(pred_Im, tar_Im)
        ImL1 = self.ImL1Loss(pred_Im, tar_Im)

        if self.masked_kspace:
            KspaceL2 = self.KspaceL2Loss(pred_K, tar_K)/(torch.sum(self.maskNot)*tar_K.max())
        else:
            KspaceL2 = self.KspaceL2Loss(pred_K, tar_K)

        fullLoss = self.ImL2_weights*ImL2 + self.ImL1_weights*ImL1 + self.KspaceL2_weights*KspaceL2
        return fullLoss, ImL2, ImL1, KspaceL2


# class netLoss(Function):
#     """Dice coeff for individual examples"""
#
#     def forward(self, input, target):
#         self.save_for_backward(input, target)
#         eps = 0.0001
#         self.inter = torch.dot(input.view(-1), target.view(-1))
#         self.union = torch.sum(input) + torch.sum(target) + eps
#
#         t = (2 * self.inter.float() + eps) / self.union.float()
#         return t
#
#     # This function has only a single output, so it gets only one gradient
#     def backward(self, grad_output):
#
#         input, target = self.saved_variables
#         grad_input = grad_target = None
#
#         if self.needs_input_grad[0]:
#             grad_input = grad_output * 2 * (target * self.union - self.inter) \
#                          / (self.union * self.union)
#         if self.needs_input_grad[1]:
#             grad_target = None
#
#         return grad_input, grad_target
#
#
# def dice_coeff(input, target):
#     """Dice coeff for batches"""
#     if input.is_cuda:
#         s = torch.FloatTensor(1).cuda().zero_()
#     else:
#         s = torch.FloatTensor(1).zero_()
#
#     for i, c in enumerate(zip(input, target)):
#         s = s + netLoss().forward(c[0], c[1])
#
#     return s / (i + 1)

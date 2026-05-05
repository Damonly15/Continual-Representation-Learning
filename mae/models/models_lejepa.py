import torch
import torch.nn as nn
import torch.distributed as dist
import timm
from torchvision.ops import MLP


class VisionTransformerLeJEPA(nn.Module):
    def __init__(self, model='vit_base_patch16_224', proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(model, pretrained=False, num_classes=512, drop_path_rate=0.1, dynamic_img_size=True)

        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17, num_slices=1024):
        super().__init__()
        self.num_slices = num_slices
        # Integrate on [0, t_max] and double via ECF symmetry (ECF(-t)=conj(ECF(t))),
        # giving effectively 2*knots-1 quadrature points at the cost of knots.
        t = torch.linspace(0, 5, knots, dtype=torch.float32)
        dt = 5 / (knots - 1)
        
        # Weights already include the factor of 2 for symmetry
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = torch.exp(-0.5 * t.square())
        
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights)

    def forward(self, proj, global_step):
        g = torch.Generator(device=proj.device)
        g.manual_seed(global_step)
        
        # Note: Added self.num_slices to make it configurable
        A = torch.randn(proj.size(-1), self.num_slices, device=proj.device, generator=g)
        A = A.div_(A.norm(p=2, dim=0))
        
        x_t = (proj @ A).unsqueeze(-1) * self.t
        ecf_real = x_t.cos().mean(-3)
        ecf_imag = x_t.sin().mean(-3)
        
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()

        ecf_combined = torch.stack([ecf_real, ecf_imag])
        ecf_combined = DifferentiableAllReduce.apply(ecf_combined)
        ecf_real, ecf_imag = ecf_combined[0], ecf_combined[1]
            
        # FIXED: Added the .mul(self.phi) window weighting
        err = ((ecf_real - self.phi).square() + ecf_imag.square()) * self.phi
        
        # FIXED: Removed the extra 2* multiplier (handled by self.weights)
        # FIXED: Multiplied local batch size by world_size for global N
        global_N = proj.size(-2) * world_size
        statistic = (err * self.weights).sum(-1) * global_N
        
        return statistic.mean()

class DifferentiableAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 1. Clone to avoid in-place modification of the autograd graph
        y = x.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(y, op=dist.ReduceOp.AVG)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # 2. The local gradient is just the incoming gradient divided by world_size
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            
        return grad_output / world_size
"""Small 2D learning-based registration experiment, explicitly designed for CPU.

B,C,H,W tensors; field channels are ux,uy in pixels. All warps map fixed
output coordinates into moving input coordinates. Truth is evaluation only.
"""
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import registration_lab as lab

ROOT = Path(__file__).resolve().parent
WEIGHT = .01
LEARNING_RATE = .001
IMAGE_SIZE = 128
MARGIN = 16
DISPLACEMENT_BOUND = 12.
MOTION_AMPLITUDE = 12.
GENERATED_CONTROL_NODES = 3
PREDICTED_CONTROL_NODES = 5
ARCHITECTURE_VERSION = 'coarse5x5_motion12_v2'


class TinyRegistrationNet(nn.Module):
    """Encoder/decoder predicting coarse control offsets, not pixel-scale offsets."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(2, 8, 3, padding=1), nn.LeakyReLU(.2),
                                     nn.Conv2d(8, 8, 3, padding=1), nn.LeakyReLU(.2))
        self.coarse = nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.LeakyReLU(.2),
                                    nn.Conv2d(16, 16, 3, padding=1), nn.LeakyReLU(.2),
                                    nn.Conv2d(16, 16, 3, padding=1), nn.LeakyReLU(.2))
        self.decoder = nn.Sequential(nn.Conv2d(24, 16, 3, padding=1), nn.LeakyReLU(.2),
                                     nn.Conv2d(16, 8, 3, padding=1), nn.LeakyReLU(.2))
        self.output = nn.Conv2d(8, 2, 3, padding=1)
        nn.init.normal_(self.output.weight, std=1e-5)
        nn.init.zeros_(self.output.bias)

    def forward(self, fixed, moving):
        fine = self.encoder(torch.cat([fixed, moving], dim=1))
        coarse = self.coarse(fine)
        coarse = F.interpolate(coarse, size=fine.shape[-2:], mode='bilinear', align_corners=True)
        features = self.decoder(torch.cat([fine, coarse], dim=1))
        controls = self.output(F.adaptive_avg_pool2d(features, (PREDICTED_CONTROL_NODES, PREDICTED_CONTROL_NODES)))
        dense = F.interpolate(controls, size=fixed.shape[-2:], mode='bicubic', align_corners=True)
        # Bound after interpolation: cubic overshoot cannot violate sampling bounds.
        return DISPLACEMENT_BOUND*torch.tanh(dense)


def warp(moving, field):
    """Differentiable bilinear backward sampling; align_corners is consistent."""
    h, w = moving.shape[-2:]
    y, x = torch.meshgrid(torch.arange(h, device=moving.device, dtype=moving.dtype),
                          torch.arange(w, device=moving.device, dtype=moving.dtype), indexing='ij')
    qx = 2*(x + field[:, 0])/(w-1)-1
    qy = 2*(y + field[:, 1])/(h-1)-1
    return F.grid_sample(moving, torch.stack([qx, qy], dim=-1), mode='bilinear',
                         padding_mode='border', align_corners=True)


def losses(fixed, moving, field, weight=WEIGHT):
    """No truth argument: training uses image agreement and field regularity only."""
    warped = warp(moving, field)
    # Bound <=12 pixels; margin 16 ensures no padded values enter this loss.
    image = (warped[:, :, MARGIN:-MARGIN, MARGIN:-MARGIN]-fixed[:, :, MARGIN:-MARGIN, MARGIN:-MARGIN]).square().mean()
    dx, dy = field[:, :, :, 1:]-field[:, :, :, :-1], field[:, :, 1:, :]-field[:, :, :-1, :]
    regularity = (dx.square().sum()+dy.square().sum())/(dx.numel()+dy.numel())
    return image + weight*regularity, image, regularity


def load_slices(split='train'):
    if split not in ['train', 'test']:
        raise ValueError('Use train or test.')
    with np.load(ROOT/'data'/'learning_ct_slices.npz', allow_pickle=False) as z:
        images = z[split].copy()
    if images.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError('Expected native 128x128 learning crops. Replace the complete practical folder with the updated bundle.')
    return torch.from_numpy(images)[:, None]


def make_batch(split='train', count=1, seed=0, index=None):
    """Same motion generator for training/evaluation; source slices are disjoint."""
    generator = torch.Generator().manual_seed(seed)
    slices = load_slices(split)
    indices = torch.randint(len(slices), (count,), generator=generator) if index is None else torch.full((count,), int(index), dtype=torch.long)
    moving = slices[indices].clone()
    coarse = (torch.rand(count, 2, GENERATED_CONTROL_NODES, GENERATED_CONTROL_NODES, generator=generator)-.5)*2
    local = F.interpolate(coarse, size=moving.shape[-2:], mode='bicubic', align_corners=True)
    translation = (torch.rand(count, 2, 1, 1, generator=generator)-.5)*2
    # Broad regional motion with smooth saturation instead of hard clipping.
    truth = MOTION_AMPLITUDE*torch.tanh(.6*local + .4*translation)
    with torch.no_grad():
        fixed = warp(moving, truth)
    return dict(fixed=fixed, moving=moving, truth=truth, indices=indices, split=split)


def new_model():
    torch.manual_seed(19)
    return TinyRegistrationNet().cpu()


def load_checkpoint(step):
    if step not in [0, 100, 1000]:
        raise ValueError('Available saved checkpoints: 0, 100, 1000 updates.')
    model = new_model()
    with np.load(ROOT/'data'/f'learning_weights_{step}.npz', allow_pickle=False) as z:
        if '__architecture__' not in z or str(z['__architecture__']) != ARCHITECTURE_VERSION:
            raise ValueError('These reference weights belong to an older model. Train the new coarse model from scratch.')
        model.load_state_dict({k: torch.from_numpy(z[k].copy()) for k in z.files if k != '__architecture__'})
    return model


def save_checkpoint(model, step):
    np.savez_compressed(ROOT/'data'/f'learning_weights_{step}.npz',
                        __architecture__=np.array(ARCHITECTURE_VERSION),
                        **{k: v.detach().cpu().numpy() for k,v in model.state_dict().items()})


def evaluate(model, batch):
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        field = model(batch['fixed'], batch['moving'])
        total, image, reg = losses(batch['fixed'], batch['moving'], field)
        warped = warp(batch['moving'], field)
        error = torch.linalg.vector_norm(field-batch['truth'], dim=1)[:, MARGIN:-MARGIN, MARGIN:-MARGIN].mean()
    fields = field.permute(0,2,3,1).numpy()
    folds = np.mean([np.mean(lab.jacobian(u)[MARGIN:-MARGIN,MARGIN:-MARGIN] <= 0)*100 for u in fields])
    return dict(total=float(total), image=float(image), regularity=float(reg),
                field_error=float(error), folds=float(folds), field=fields[0],
                warped=warped[0,0].numpy(), seconds=time.perf_counter()-start)


def train_update(model, optimizer, batch, weight=WEIGHT, inspect=False):
    """One weight update; optional gradient evidence for the sampler and network."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    before = {k: p.detach().clone() for k,p in model.named_parameters()} if inspect else None
    field = model(batch['fixed'], batch['moving'])
    if inspect:
        field.retain_grad()
    total, image, regularity = losses(batch['fixed'], batch['moving'], field, weight)
    image_gradient = torch.autograd.grad(image, field, retain_graph=True)[0] if inspect else None
    total.backward()
    info = dict(total=float(total.detach()), image=float(image.detach()), regularity=float(regularity.detach()))
    if inspect:
        info['field_gradient'] = image_gradient[0].norm(dim=0).detach().numpy()
        info['network_gradient_norm'] = float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)))
    optimizer.step()
    if inspect:
        info['weight_change_norm'] = float(torch.sqrt(sum((p.detach()-before[k]).square().sum() for k,p in model.named_parameters())))
    return info


def training_block(model, optimizer, steps=20, start_step=0):
    if not isinstance(steps, int) or steps < 1 or steps > 1000:
        raise ValueError('Choose 1 to 1000 updates per block.')
    rows = []
    for step in range(start_step, start_step+steps):
        batch = make_batch('train', count=8, seed=1000+step)
        info = train_update(model, optimizer, batch)
        rows.append(dict(step=step+1, **info))
    return rows


def plot_result(batch, result, title):
    fixed, moving = batch['fixed'][0,0].numpy(), batch['moving'][0,0].numpy()
    fig, axes = plt.subplots(3,3,figsize=(12,10))
    mask = np.zeros(fixed.shape, dtype=bool); mask[MARGIN:-MARGIN,MARGIN:-MARGIN] = True
    lab.deformation_panels(axes, fixed, moving, result['warped'], result['field'], mask, grid_step=8)
    fig.suptitle(f'{title}\nImage MSE {result["image"]:.5f}; field error {result["field_error"]:.3f} px; nonpositive Jacobians {result["folds"]:.2f}%\nOverlays: cyan = target, red = moving/registered. Differences share one signed scale; gray = outside ROI.', fontsize=10)
    fig.tight_layout(rect=(0,0,1,.94));return fig


def plot_loss_parts(result, weight=WEIGHT):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(['Image mismatch S','Weighted penalty λR'], [result['image'], weight*result['regularity']])
    ax.set(ylabel='Contribution to training objective', title=f'Total S + λR = {result["total"]:.6f}')
    fig.tight_layout();return fig


def plot_gradient(batch, info):
    fig, axes = plt.subplots(1,2,figsize=(7,3))
    axes[0].imshow(batch['moving'][0,0],cmap='gray');axes[0].set_title('Moving CT')
    im=axes[1].imshow(info['field_gradient'],cmap='magma');axes[1].set_title('Image-loss gradient magnitude\nwith respect to displacement', fontsize=10)
    fig.colorbar(im,ax=axes[1]);fig.tight_layout();return fig

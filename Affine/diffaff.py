# Making PyTorch's torchvision.transforms.functional.affine differentiable
from re import S
import torch, torchviz
from torch import Tensor
from typing import List, Tuple
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F

pil_modes_mapping = {
    InterpolationMode.NEAREST: 0,
    InterpolationMode.BILINEAR: 2,
    InterpolationMode.BICUBIC: 3,
    InterpolationMode.BOX: 4,
    InterpolationMode.HAMMING: 5,
    InterpolationMode.LANCZOS: 1,
}

def _get_inverse_affine_matrix(
    center, angle, translate, scale, shear, inverted: bool = True
):
    rot = angle
    sx = shear[0]
    sy = shear[1]

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = torch.cos(rot - sy) / torch.cos(sy)
    b = -torch.cos(rot - sy) * torch.tan(sx) / torch.cos(sy) - torch.sin(rot)
    c = torch.sin(rot - sy) / torch.cos(sy)
    d = -torch.sin(rot - sy) * torch.tan(sx) / torch.cos(sy) + torch.cos(rot)

    # Inverted rotation matrix with scale and shear
    # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
    matrix = torch.stack([d / scale,
                            -b / scale,
                            d / scale * (-cx - tx) + -b / scale * (-cy - ty) + cx,
                            -c / scale,
                            a / scale,
                            -c / scale * (-cx - tx) +  a / scale * (-cy - ty) + cy
                            ])
    return matrix

def _cast_squeeze_in(img, req_dtypes: List[torch.dtype]) -> Tuple[Tensor, bool, bool, torch.dtype]: # hmmmmmm
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype

def _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype: torch.dtype) -> Tensor:
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            # it is better to round before cast
            img = torch.round(img)
        img = img.to(out_dtype)

    return img

def _gen_affine_grid( # hmmmmmmmm
    theta,
    w,
    h,
    ow,
    oh,
) -> Tensor:
    d = 0.5
    base_grid = torch.empty(1, oh, ow, 3, dtype=theta.dtype, device=theta.device)
    x_grid = torch.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, steps=ow, device=theta.device)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, steps=oh, device=theta.device).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)

    rescaled_theta = theta.transpose(1, 2) / torch.tensor([0.5 * w, 0.5 * h], dtype=theta.dtype, device=theta.device)
    output_grid = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta)
    return output_grid.view(1, oh, ow, 2)

def _apply_grid_transform(img, grid, mode: str, fill) -> Tensor:
    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [grid.dtype])

    if img.shape[0] > 1: grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])
    
    img = F.grid_sample(img, grid, mode=mode, padding_mode="zeros", align_corners=False)
    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    """torchviz.make_dot(grid).render("zimg_grid",format="png")
    torchviz.make_dot(img).render("zimg_img",format="png")"""
    return img

def apply_affine(
    img, matrix, interpolation: str = "nearest", fill = None
) -> Tensor:
    theta = matrix.view(1, 2, 3)
    shape = img.shape
    # grid will be generated on the same device as theta and img
    grid = _gen_affine_grid(theta, w=shape[-1], h=shape[-2], ow=shape[-1], oh=shape[-2])
    return _apply_grid_transform(img, grid, interpolation, fill=fill)

def affine(
    img,
    angle,
    translate,
    scale,
    shear,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill = None,
    center = None,
) -> Tensor:
    height, width = img.shape[-2:]
    center_f = [0.0, 0.0]
    if center is not None:
        height, width = img.shape[-2:]
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, [width, height])]

    matrix = _get_inverse_affine_matrix(center_f, angle, translate, scale, shear)
    
    return apply_affine(img, matrix=matrix, interpolation=interpolation.value, fill=fill)

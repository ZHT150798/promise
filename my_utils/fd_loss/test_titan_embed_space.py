import os
import sys

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import h5py

sys.path.append(os.getcwd())
import torch

from PIL import Image
import torch.nn as nn
from my_utils.fd_loss.conch import build_conch
from my_utils.fd_loss import pe_loss
import torchvision.transforms as T
Image.MAX_IMAGE_PIXELS = None

def extract_patches(image, coords, patch_size=512):
    patches = []
    for (x, y) in coords:
        patch = image.crop((x, y, x+patch_size, y+patch_size))
        patches.append(patch)
    return patches

img_size = 448
eval_transform = T.Compose([
    T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(img_size),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    demo_h5_path = "/preset/datasets/tcga/test/seg_s/features/h5_files/TCGA-55-A491-01Z-00-DX1.E5F3B4E5-18EA-4067-AE28-119DABCE739A.h5"
    file = h5py.File(demo_h5_path, 'r')
    features = torch.from_numpy(file['features'][:]).unsqueeze(0)
    coords_tensor = torch.from_numpy(file['coords'][:]).to(torch.int64)
    coords_list = coords_tensor.cpu().numpy().tolist()

    conch, _ = build_conch()
    conch.requires_grad_(False)
    pe_loss_fn = pe_loss.PELoss(conch, eval_transform).to(device)

    img = Image.open(
        "/preset/datasets/tcga/test/slide_s/sr_v2_pe/TCGA-55-A491-01Z-00-DX1.E5F3B4E5-18EA-4067-AE28-119DABCE739A.png").convert("RGB")
    img_hr = Image.open(
        "/preset/datasets/tcga/test/slide_s/hr/TCGA-55-A491-01Z-00-DX1.E5F3B4E5-18EA-4067-AE28-119DABCE739A.png").convert("RGB")
    patch = extract_patches(img, coords_list, 1024)[10]
    patch_hr = extract_patches(img_hr, coords_list, 1024)[10]

    patch_hr = eval_transform(patch_hr).unsqueeze(0).to(device)
    patch = eval_transform(patch).unsqueeze(0).to(device)

    hr_feats = conch(patch_hr)
    sr_feats = conch(patch)
    true_feats = features.squeeze(0)[10]

    print(nn.functional.mse_loss(hr_feats.cpu(), sr_feats.cpu(), reduction='mean'))
    print(nn.functional.mse_loss(true_feats.cpu(), hr_feats.cpu(), reduction='mean'))

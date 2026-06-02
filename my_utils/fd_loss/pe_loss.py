import os

# Pathology feature losses used by training and evaluation.
import torch
import timm

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from my_utils.fd_loss.conch import build_conch


def tensor_to_numpy_img(tensor):
    """
    将 [C, H, W] 的 tensor 转换为 numpy 数组，并转置为 [H, W, C] 格式，归一化到 [0,1] 范围
    """
    # 如果 tensor 是浮点型，确保在 [0,1] 范围；如果不是可以做相应归一化
    np_img = tensor.detach().cpu().squeeze(0).numpy()
    # 转换通道位置：[C, H, W] -> [H, W, C]
    np_img = np.transpose(np_img, (1, 2, 0))
    # 如果数值范围在[0,1]则直接返回，否则需要归一化或转换数据类型
    return np.clip(np_img, 0, 1)


class PELoss(nn.Module):
    def __init__(self, conch):
        super(PELoss, self).__init__()
        self.conch = conch
        self.eval_transform = transforms.Compose([
            transforms.Resize(448, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(448),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        #self.model.eval()
        self.conch.requires_grad_(False)

    def compute_feat_diff(self, sr, hr):
        """
        计算两个 batch 之间的 L2 损失。

        :param img1_tensor_batch: 第一个 batch 的图像 tensor，形状为 (B, C, H, W)
        :param img2_tensor_batch: 第二个 batch 的图像 tensor，形状为 (B, C, H, W)
        :return: L2 损失
        """

        # 计算两个 batch 的特征
        sr_feat = self.conch(sr).squeeze()
        hr_feat = self.conch(hr).squeeze()

        # 计算每个图像对的 L1
        l1_loss = nn.functional.l1_loss(sr_feat, hr_feat, reduction='mean')

        return l1_loss, sr_feat, hr_feat

    def forward(self, sr, hr, mag_prompt="40× magnification", pad_size=1024):
        """
        计算两个 batch 之间的 L2 损失。

        :param img1_tensor_batch: 第一个 batch 的图像 tensor，形状为 (B, C, H, W)
        :param img2_tensor_batch: 第二个 batch 的图像 tensor，形状为 (B, C, H, W)
        :return: L2 损失
        """
        # (-1, 1)--(0, 1)
        sr = sr * 0.5 + 0.5
        hr = hr * 0.5 + 0.5
        # 40ximg padding to (1024*1024)
        _, _, H, W = hr.shape
        if isinstance(mag_prompt, list):
            mag_prompt = mag_prompt[0]
        if H != 1024 and mag_prompt == "40× magnification":
            pad = max(pad_size - H, 0)
            # 将填充均匀分配到两侧，如果填充量为奇数，右边/下边多填 1
            pad_top = pad // 2
            pad_bottom = pad - pad_top
            # sr = F.pad(sr, (pad_top, pad_bottom, pad_top, pad_bottom), mode='constant', value=0)
            # hr = F.pad(hr, (pad_top, pad_bottom, pad_top, pad_bottom), mode='constant', value=0)
            sr = F.pad(sr, (pad_top, pad_bottom, pad_top, pad_bottom), mode='reflect')
            hr = F.pad(hr, (pad_top, pad_bottom, pad_top, pad_bottom), mode='reflect')
        # # 可视化原始和填充后的图像
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # axs[0].imshow(original_img)
        # axs[0].set_title("Original Image")
        # axs[0].axis("off")
        #
        # axs[1].imshow(padded_img)
        # axs[1].set_title("Padded Image")
        # axs[1].axis("off")
        #
        # plt.show()

        # 计算两个 batch 的特征
        sr_feat = self.conch(self.eval_transform(sr)).squeeze()
        hr_feat = self.conch(self.eval_transform(hr)).squeeze()

        # 计算每个图像对的 L1 损失
        l1_loss = nn.functional.l1_loss(sr_feat, hr_feat, reduction='mean')

        return l1_loss


class PELoss_uni(nn.Module):
    def __init__(self, model):
        super(PELoss_uni, self).__init__()
        self.model = model
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, sr, hr):
        """
        计算两个 batch 之间的 L2 损失。

        :param img1_tensor_batch: 第一个 batch 的图像 tensor，形状为 (B, C, H, W)
        :param img2_tensor_batch: 第二个 batch 的图像 tensor，形状为 (B, C, H, W)
        :return: L2 损失
        """
        # (-1, 1)--(0, 1)
        sr = sr * 0.5 + 0.5
        hr = hr * 0.5 + 0.5
        # 40ximg padding to (1024*1024)
        _, _, H, W = hr.shape


        # 计算两个 batch 的特征
        sr_feat = self.model(self.eval_transform(sr)).squeeze()
        hr_feat = self.model(self.eval_transform(hr)).squeeze()

        # 计算每个图像对的 L1 损失
        l1_loss = nn.functional.l1_loss(sr_feat, hr_feat, reduction='mean')

        return l1_loss


class PELoss_prov(nn.Module):
    def __init__(self, model):
        super(PELoss_prov, self).__init__()
        self.model = model
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, sr, hr):
        """
        计算两个 batch 之间的 L2 损失。

        :param img1_tensor_batch: 第一个 batch 的图像 tensor，形状为 (B, C, H, W)
        :param img2_tensor_batch: 第二个 batch 的图像 tensor，形状为 (B, C, H, W)
        :return: L2 损失
        """
        # (-1, 1)--(0, 1)
        sr = sr * 0.5 + 0.5
        hr = hr * 0.5 + 0.5

        _, _, H, W = hr.shape
        # 计算两个 batch 的特征
        sr_feat = self.model(self.eval_transform(sr)).squeeze()
        hr_feat = self.model(self.eval_transform(hr)).squeeze()

        # 计算每个图像对的 L1 损失
        l1_loss = nn.functional.l1_loss(sr_feat, hr_feat, reduction='mean')

        return l1_loss


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    # 假设 conch 和 eval_transform 已经初始化
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False,
                              checkpoint_path="/media/dell/data/zhangv1/WSISR/huggingface/prov-gigapath/pytorch_model.bin")

    # 创建 PELoss 实例

    pe_loss_fn = PELoss_prov(model.cuda())

    # 假设 img1_tensor_batch 和 img2_tensor_batch 是你的输入的两个 batch tensor
    hr = torch.randn(1, 3, 512, 512).cuda()
    lr = F.interpolate(hr, size=(128, 128), mode='bilinear')
    sr = F.interpolate(lr, size=(512, 512), mode='bicubic')

    a = pe_loss_fn(sr, hr)
    print(a)
    # # 输出损失
    # print(f"PE Loss: {pe_loss.item():.6f}")

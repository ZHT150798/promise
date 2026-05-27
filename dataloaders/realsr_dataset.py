import os
import glob
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
import torchvision.transforms.functional as F
from dataloaders.realesrgan import RealESRGAN_degradation

cancer_infor = {
    "GBM": {"prompt": "Glioblastoma Multiforme, Brain"},
    "HCC": {"prompt": "Hepatocellular Carcinoma, Liver"},
    "UCEC": {"prompt": "Uterine Corpus Endometrial Carcinoma, Uterus"},
    "THCA": {"prompt": "Thyroid Carcinoma, Thyroid"},
    "KIRC": {"prompt": "Kidney Renal Clear Cell Carcinoma, Kidney"},
    "LUSC": {"prompt": "Lung Squamous Cell Carcinoma, Lung"},
    "LUAD": {"prompt": "Lung Adenocarcinoma, Lung"},
    "BRCA": {"prompt": "Breast Invasive Carcinoma, Breast"},
    "CRC_READ": {"prompt": "Rectum Adenocarcinoma, Rectum"},
    "CRC_COAD": {"prompt": "Colon Adenocarcinoma, Colon"},
    "HNSC": {"prompt": "Head and Neck Squamous Cell Carcinoma, Head and Neck"}
}


class PairedSRPathoDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()
        self.args = args
        self.split = split
        if split == 'train':
            self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((512, 512)),
                transforms.RandomHorizontalFlip(),
            ])

            self.gt_list = []
            assert len(args.dataset_txt_paths_list) == len(args.dataset_prob_paths_list)
            for idx_dataset in range(len(args.dataset_txt_paths_list)):
                with open(args.dataset_txt_paths_list[idx_dataset], 'r') as f:
                    dataset_list = [line.strip() for line in f.readlines()]
                    for idx_ratio in range(args.dataset_prob_paths_list[idx_dataset]):
                        gt_length = len(self.gt_list)
                        self.gt_list += dataset_list
                        print(f'=====> append {len(self.gt_list) - gt_length} data.')

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        if self.split == 'train':
            gt_img = Image.open(self.gt_list[idx]).convert('RGB')
            gt_img = self.crop_preproc(gt_img)

            output_t, img_t = self.degradation.degrade_process(np.asarray(gt_img) / 255., resize_bak=True)
            output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)

            # input images scaled to -1,1
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            # example["prompt"] = caption
            example["neg_prompt"] = self.args.neg_prompt
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t

            return example


class PairedSRPathoDataset_test(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()
        self.args = args
        self.split = split
        if split == 'test':
            self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
            self.gt_list = []
            assert len(args.dataset_txt_paths_list_test) == len(args.dataset_prob_paths_list)
            for idx_dataset in range(len(args.dataset_txt_paths_list_test)):
                with open(args.dataset_txt_paths_list_test[idx_dataset], 'r') as f:
                    dataset_list = [line.strip() for line in f.readlines()]
                    for idx_ratio in range(args.dataset_prob_paths_list[idx_dataset]):
                        gt_length = len(self.gt_list)
                        self.gt_list += dataset_list
                        print(f'=====> append {len(self.gt_list) - gt_length} data.')

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        if self.split == 'test':
            gt_img = Image.open(self.gt_list[idx]).convert('RGB')

            output_t, img_t = self.degradation.degrade_process(np.asarray(gt_img) / 255., resize_bak=True)
            output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)

            # input images scaled to -1,1
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            # example["prompt"] = caption
            example["neg_prompt"] = self.args.neg_prompt
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t

            return example


class WSISRDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()
        self.args = args
        self.split = split
        self.gt_path = args.train_path + "hr/"
        self.lr_path = args.train_path + "lr/"
        self.deg_type = args.deg_type
        self.cancer_infor = cancer_infor
        self.sr_prompt = None

        if split == 'train':
            self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((512, 512)),
                transforms.RandomHorizontalFlip(),
            ])
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(),
            ])

            self.gt_list, self.lr_list = [], []
            self.gt_list.extend(sorted([str(x) for x in Path(self.gt_path).rglob('*.png')]))
            if self.deg_type == "lr":
                self.lr_list.extend(sorted([str(x) for x in Path(self.lr_path).rglob('*.png')]))
            print(f'=====> append {len(self.gt_list)} data.')
            # assert len(args.dataset_txt_paths_list) == len(args.dataset_prob_paths_list)
            # for idx_dataset in range(len(args.dataset_txt_paths_list)):
            #     with open(args.dataset_txt_paths_list[idx_dataset], 'r') as f:
            #         dataset_list = [line.strip() for line in f.readlines()]
            #         for idx_ratio in range(args.dataset_prob_paths_list[idx_dataset]):
            #             gt_length = len(self.gt_list)
            #             self.gt_list += dataset_list
            #             print(f'=====> append {len(self.gt_list) - gt_length} data.')

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        if self.split == 'train':
            img_name = self.gt_list[idx].split("/")[-1]
            gt_img = Image.open(self.gt_path + img_name).convert('RGB')
            #gt_img = self.crop_preproc(gt_img)
            if self.deg_type == "realgan":
                output_t, img_t = self.degradation.degrade_process(np.asarray(gt_img) / 255., resize_bak=True)
                output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)
            elif self.deg_type == "lr":
                lr_img = Image.open(self.lr_path + img_name).convert('RGB')
                output_t, img_t = self.degradation.random_augment_pair(np.asarray(gt_img) / 255.,np.asarray(lr_img) / 255.)
                output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)
            elif self.deg_type == "compress":
                lr_img = Image.open(self.lr_path + img_name).convert('RGB')
                output_t, img_t, sr_prompt = self.degradation.random_augment_lr_compress(np.asarray(gt_img) / 255.,np.asarray(lr_img) / 255.)
                output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)
                self.sr_prompt = sr_prompt
            elif self.deg_type == "norm":
                output_t, img_t = self.degradation.random_augment_norm(np.asarray(gt_img) / 255.)
                output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)
            else:
                raise ValueError(f"Incorrect deg type {self.deg_type} (norm, realgan, lr, compress).")
            # input images scaled to -1,1
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            # example["prompt"] = caption
            if self.sr_prompt:
                example["neg_prompt"] = self.sr_prompt
                example["pos_prompt"] = self.sr_prompt
            else:
                example["neg_prompt"] = ""
                example["pos_prompt"] = ""
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t

            return example


class WSISRDataset_val(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()
        self.args = args
        self.split = split
        self.gt_path = args.val_path + "hr/"
        self.lr_path = args.val_path + "lr/"
        self.deg_type = args.deg_type
        self.sr_prompt = None
        if split == 'test':
            self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(),
            ])

            self.gt_list, self.lr_list = [], []
            self.gt_list.extend(sorted([str(x) for x in Path(self.gt_path).rglob('*.png')]))
            # if self.deg_type == "lr":
            #     self.lr_list.extend(sorted([str(x) for x in Path(self.lr_path).rglob('*.png')]))
            print(f'=====> append {len(self.gt_list)} data.')
            # assert len(args.dataset_txt_paths_list) == len(args.dataset_prob_paths_list)
            # for idx_dataset in range(len(args.dataset_txt_paths_list)):
            #     with open(args.dataset_txt_paths_list[idx_dataset], 'r') as f:
            #         dataset_list = [line.strip() for line in f.readlines()]
            #         for idx_ratio in range(args.dataset_prob_paths_list[idx_dataset]):
            #             gt_length = len(self.gt_list)
            #             self.gt_list += dataset_list
            #             print(f'=====> append {len(self.gt_list) - gt_length} data.')

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        if self.split == 'test':
            img_name = self.gt_list[idx].split("/")[-1]
            gt_img = Image.open(self.gt_path+img_name).convert('RGB')
            p = 1
            #gt_img = self.crop_preproc(gt_img)
            if self.deg_type == "realgan":
                output_t, img_t = self.degradation.degrade_process(np.asarray(gt_img) / 255., resize_bak=True)
                output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)
            elif self.deg_type == "lr":
                lr_img = Image.open(self.lr_path + img_name).convert('RGB')
                output_t, img_t, _ = self.degradation.random_augment_pair_infer(np.asarray(gt_img) / 255.,np.asarray(lr_img) / 255.)
                output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)
            elif self.deg_type == "compress":
                lr_img = Image.open(self.lr_path + img_name).convert('RGB')
                output_t, img_t, sr_prompt = self.degradation.random_augment_pair_infer(np.asarray(gt_img) / 255.,np.asarray(lr_img) / 255.)
                output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)
                self.sr_prompt = sr_prompt
            elif self.deg_type == "norm":
                output_t, img_t = self.degradation.random_augment_norm(np.asarray(gt_img) / 255.)
                output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)
            else:
                raise ValueError(f"Incorrect deg type {self.deg_type} (norm, realgan, lr).")
            # input images scaled to -1,1
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            # example["prompt"] = caption
            if self.sr_prompt:
                example["neg_prompt"] = self.sr_prompt
                example["pos_prompt"] = self.sr_prompt
            else:
                example["neg_prompt"] = ""
                example["pos_prompt"] = ""
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t

            return example

class WSISRDataset_test(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()
        self.args = args
        self.split = split
        self.gt_path = args.val_path + "hr/"
        self.lr_path = args.val_path + "lr/"
        self.deg_type = args.deg_type
        if split == 'test':
            self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(),
            ])

            self.gt_list, self.lr_list = [], []
            self.gt_list.extend(sorted([str(x) for x in Path(self.gt_path).rglob('*.png')]))
            # if self.deg_type == "lr":
            #     self.lr_list.extend(sorted([str(x) for x in Path(self.lr_path).rglob('*.png')]))
            print(f'=====> append {len(self.gt_list)} data.')
            # assert len(args.dataset_txt_paths_list) == len(args.dataset_prob_paths_list)
            # for idx_dataset in range(len(args.dataset_txt_paths_list)):
            #     with open(args.dataset_txt_paths_list[idx_dataset], 'r') as f:
            #         dataset_list = [line.strip() for line in f.readlines()]
            #         for idx_ratio in range(args.dataset_prob_paths_list[idx_dataset]):
            #             gt_length = len(self.gt_list)
            #             self.gt_list += dataset_list
            #             print(f'=====> append {len(self.gt_list) - gt_length} data.')

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        if self.split == 'test':
            img_name = self.gt_list[idx].split("/")[-1]
            gt_img = Image.open(self.gt_path+img_name).convert('RGB')
            #gt_img = self.crop_preproc(gt_img)
            lr_img = Image.open(self.lr_path + img_name).convert('RGB')

            output_t, img_t = self.degradation.random_augment_pair_infer(np.asarray(gt_img) / 255.,np.asarray(lr_img) / 255.)
            output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)

            # input images scaled to -1,1
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            # example["prompt"] = caption
            example["neg_prompt"] = self.args.neg_prompt
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t

            return example


class WSISRDataset_val_compress(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None, scale = 4):
        super().__init__()
        self.args = args
        self.split = split
        if scale !=4:
            self.gt_path = args.val_path + "hr/"
            self.lr_path = args.val_path + f"lr_X{scale}/"
        else:
            self.gt_path = args.val_path + "hr/"
            self.lr_path = args.val_path + "lr/"
        self.deg_type = args.deg_type
        self.sr_prompt = None
        if split == 'test':
            self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(),
            ])

            self.gt_list, self.lr_list = [], []
            self.gt_list.extend(sorted([str(x) for x in Path(self.gt_path).rglob('*.png')]))
            # if self.deg_type == "lr":
            #     self.lr_list.extend(sorted([str(x) for x in Path(self.lr_path).rglob('*.png')]))
            print(f'=====> append {len(self.gt_list)} data.')
            # assert len(args.dataset_txt_paths_list) == len(args.dataset_prob_paths_list)
            # for idx_dataset in range(len(args.dataset_txt_paths_list)):
            #     with open(args.dataset_txt_paths_list[idx_dataset], 'r') as f:
            #         dataset_list = [line.strip() for line in f.readlines()]
            #         for idx_ratio in range(args.dataset_prob_paths_list[idx_dataset]):
            #             gt_length = len(self.gt_list)
            #             self.gt_list += dataset_list
            #             print(f'=====> append {len(self.gt_list) - gt_length} data.')

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        if self.split == 'test':
            img_name = self.gt_list[idx].split("/")[-1]
            gt_img = Image.open(self.gt_path+img_name).convert('RGB')
            p = 1
            #gt_img = self.crop_preproc(gt_img)
            if self.deg_type == "realgan":
                output_t, img_t = self.degradation.degrade_process(np.asarray(gt_img) / 255., resize_bak=True)
                output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)
            elif self.deg_type == "lr":
                lr_img = Image.open(self.lr_path + img_name).convert('RGB')
                output_t, img_t, _ = self.degradation.random_augment_pair_infer(np.asarray(gt_img) / 255.,np.asarray(lr_img) / 255.)
                output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)
            elif self.deg_type == "compress":
                #lr_img = Image.open(self.lr_path + img_name).convert('RGB')
                img_base = os.path.splitext(img_name)[0]  # 去掉扩展名
                png_path = os.path.join(self.lr_path, img_base + '.png')
                jpg_path = os.path.join(self.lr_path, img_base + '.jpg')
                if os.path.exists(png_path):
                    lr_img = Image.open(png_path).convert('RGB')
                elif os.path.exists(jpg_path):
                    lr_img = Image.open(jpg_path).convert('RGB')
                else:
                    raise FileNotFoundError(f"Neither {png_path} nor {jpg_path} exists.")
                output_t, img_t, sr_prompt = self.degradation.random_augment_pair_infer(np.asarray(gt_img) / 255.,np.asarray(lr_img) / 255.)
                output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)
                self.sr_prompt = sr_prompt
            elif self.deg_type == "norm":
                output_t, img_t = self.degradation.random_augment_norm(np.asarray(gt_img) / 255.)
                output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)
            else:
                raise ValueError(f"Incorrect deg type {self.deg_type} (norm, realgan, lr).")
            # input images scaled to -1,1
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            # example["prompt"] = caption
            if self.sr_prompt:
                example["neg_prompt"] = self.sr_prompt
                example["pos_prompt"] = self.sr_prompt
            else:
                example["neg_prompt"] = ""
                example["pos_prompt"] = ""
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t

            return example


class WSISRDataset_test_compress(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()
        self.args = args
        self.split = split
        self.gt_path = args.val_path + "hr/"
        self.lr_path = args.val_path + "lr/"
        self.deg_type = args.deg_type
        self.sr_prompt = None
        if split == 'test':
            self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(),
            ])

            self.gt_list, self.lr_list = [], []
            self.gt_list.extend(sorted([str(x) for x in Path(self.gt_path).rglob('*.png')]))
            # if self.deg_type == "lr":
            #     self.lr_list.extend(sorted([str(x) for x in Path(self.lr_path).rglob('*.png')]))
            print(f'=====> append {len(self.gt_list)} data.')
            # assert len(args.dataset_txt_paths_list) == len(args.dataset_prob_paths_list)
            # for idx_dataset in range(len(args.dataset_txt_paths_list)):
            #     with open(args.dataset_txt_paths_list[idx_dataset], 'r') as f:
            #         dataset_list = [line.strip() for line in f.readlines()]
            #         for idx_ratio in range(args.dataset_prob_paths_list[idx_dataset]):
            #             gt_length = len(self.gt_list)
            #             self.gt_list += dataset_list
            #             print(f'=====> append {len(self.gt_list) - gt_length} data.')

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        if self.split == 'test':
            img_name = self.gt_list[idx].split("/")[-1]
            gt_img = Image.open(self.gt_path+img_name).convert('RGB')
            #gt_img = self.crop_preproc(gt_img)
            lr_img = Image.open(self.lr_path + img_name).convert('RGB')

            output_t, img_t, sr_prompt = self.degradation.random_augment_pair_infer(np.asarray(gt_img) / 255.,np.asarray(lr_img) / 255.)
            output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)
            self.sr_prompt = sr_prompt

            # input images scaled to -1,1
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            # example["prompt"] = caption
            if self.sr_prompt:
                example["neg_prompt"] = self.sr_prompt
                example["pos_prompt"] = self.sr_prompt
            else:
                example["neg_prompt"] = ""
                example["pos_prompt"] = ""
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t

            return example

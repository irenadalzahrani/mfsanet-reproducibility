import glob, io, os, random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from config import PATHS, TRAIN

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def _jpeg_compress(img, quality):
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=quality); buf.seek(0)
    return Image.open(buf).convert("RGB")

def _gaussian_noise(img, sigma):
    arr = np.asarray(img).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 1) * 255.0
    return Image.fromarray(arr.astype(np.uint8))

class FreqAwareTransform:
    def __init__(self, split, image_size):
        self.split = split
        if split == "train":
            self.spatial = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=TRAIN.hflip_p),
                transforms.RandomVerticalFlip(p=TRAIN.vflip_p),
                transforms.RandomRotation(TRAIN.rotation_deg),
                transforms.RandomGrayscale(p=TRAIN.grayscale_p),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        else:
            self.spatial = transforms.Compose([
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
            ])
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __call__(self, img):
        img = self.spatial(img)
        freq_flag = 0
        if self.split == "train":
            r = random.random()
            if r < TRAIN.jpeg_aug_p:
                img = _jpeg_compress(img, random.randint(*TRAIN.jpeg_quality_range)); freq_flag = 1
            elif r < TRAIN.jpeg_aug_p + TRAIN.gaussian_noise_p:
                img = _gaussian_noise(img, random.uniform(*TRAIN.gaussian_noise_sigma_range)); freq_flag = 1
        return self.to_tensor(img), freq_flag

class DeepfakeFrameDataset(Dataset):
    def __init__(self, dataset, split, image_size=None):
        self.dataset, self.split = dataset, split
        self.image_size = image_size or TRAIN.image_size
        self.transform = FreqAwareTransform(split, self.image_size)
        self.samples = []
        for label_name, label_int in (("real", 1), ("fake", 0)):
            d = os.path.join(PATHS.frames_root, dataset, split, label_name)
            for p in sorted(glob.glob(os.path.join(d, "*.jpg"))):
                video_id = os.path.basename(p).split("__f")[0]
                self.samples.append((p, label_int, video_id))
        if not self.samples:
            raise RuntimeError(f"No frames for dataset={dataset} split={split}. Run frame extraction first.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label, video_id = self.samples[idx]
        img = Image.open(path).convert("RGB")
        tensor, freq_flag = self.transform(img)
        return tensor, label, freq_flag, video_id

    def sample_weights_for_balancing(self):
        return [TRAIN.fake_sample_weight if lbl == 0 else 1.0 for (_, lbl, _) in self.samples]

class MixedDomainDataset(Dataset):
    def __init__(self, image_size=None):
        from config import SPLITS
        self.image_size = image_size or TRAIN.image_size
        self.transform = FreqAwareTransform("train", self.image_size)
        self.samples = []
        self.samples += DeepfakeFrameDataset("ffpp", "train", self.image_size).samples
        for extra_ds, frac in (("celebdf", SPLITS.celebdf_train_frac_for_da),
                                ("dfdc", SPLITS.dfdc_train_frac_for_da)):
            extra = DeepfakeFrameDataset(extra_ds, "train", self.image_size)
            by_video = {}
            for s in extra.samples: by_video.setdefault(s[2], []).append(s)
            vids = sorted(by_video.keys())
            rng = random.Random(1234); rng.shuffle(vids)
            for vid in vids[: int(len(vids) * frac)]:
                self.samples += by_video[vid]

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label, video_id = self.samples[idx]
        img = Image.open(path).convert("RGB")
        tensor, freq_flag = self.transform(img)
        return tensor, label, freq_flag, video_id
    def sample_weights_for_balancing(self):
        return [TRAIN.fake_sample_weight if lbl == 0 else 1.0 for (_, lbl, _) in self.samples]

def make_weighted_sampler(ds):
    w = ds.sample_weights_for_balancing()
    return WeightedRandomSampler(w, num_samples=len(w), replacement=True)

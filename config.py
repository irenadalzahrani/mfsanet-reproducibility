
from dataclasses import dataclass, field
from typing import List

@dataclass
class Paths:
    ffpp_root: str = "/workspace/data/raw/ffpp/FaceForensics++_C23"
    celebdf_root: str = "/workspace/data/raw/celebdf"
    dfdc_part_dirs: List[str] = field(default_factory=lambda: ['/workspace/data/raw/dfdc/dfdc_train_part_01/dfdc_train_part_1', '/workspace/data/raw/dfdc/dfdc_train_part_02/dfdc_train_part_2', '/workspace/data/raw/dfdc/dfdc_train_part_03/dfdc_train_part_3', '/workspace/data/raw/dfdc/dfdc_train_part_04/dfdc_train_part_4'])

    frames_root: str = "/workspace/data/frames"
    splits_root: str = "/workspace/data/splits"
    ckpt_root: str = "/workspace/runs"
    results_root: str = "/workspace/results"

@dataclass
class Preprocess:
    face_size: int = 288
    bbox_margin_px: int = 20
    frames_per_video: int = 20
    fallback_full_frame: bool = True

@dataclass
class Splits:
    ffpp_compression: str = "c23"
    celebdf_val_fraction_of_trainval: float = 0.10
    celebdf_train_frac_for_da: float = 0.30
    dfdc_train_frac_for_da: float = 0.20

@dataclass
class Training:
    image_size: int = 288
    batch_size_per_step: int = 16
    grad_accum_steps: int = 2
    base_lr: float = 2e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    max_epochs: int = 60
    eta_min: float = 1e-6
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 15
    label_smoothing: float = 0.1
    fake_class_weight: float = 2.0
    fake_sample_weight: float = 2.0
    mixup_alpha: float = 0.4
    mixup_prob: float = 0.4
    threshold_fixed: float = 0.5
    threshold_calibrated: float = 0.35
    selection_auc_weight: float = 0.6
    selection_f1_weight: float = 0.4
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    hflip_p: float = 0.5
    vflip_p: float = 0.1
    rotation_deg: float = 15.0
    grayscale_p: float = 0.05
    jpeg_aug_p: float = 0.6
    jpeg_quality_range: tuple = (40, 85)
    gaussian_noise_p: float = 0.3
    gaussian_noise_sigma_range: tuple = (0.01, 0.08)
    freq_lambda: float = 0.5
    freq_weight: float = 4.0

@dataclass
class Robustness:
    jpeg_qualities: List[int] = field(default_factory=lambda: [90, 70, 50, 30])
    gaussian_sigmas: List[float] = field(default_factory=lambda: [0.02, 0.04, 0.06, 0.08])

PATHS = Paths()
PREP = Preprocess()
SPLITS = Splits()
TRAIN = Training()
ROBUST = Robustness()

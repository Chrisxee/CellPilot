from torch.utils.data import Dataset, DataLoader, random_split
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from .data_fetching import DataFetcher
from .data_utils import DataProcessing

class HistologyDataset(Dataset):
    def __init__(self, datasets, data_directory, cluster, image_encoder_size, mask_augmentation_tries, data_augmentations, threshold_connected_components):
        self.datasets = datasets
        self.data_fetcher = DataFetcher(data_directory, cluster)
        self.image_encoder_size = image_encoder_size
        self.mask_augmentation_tries = mask_augmentation_tries
        self.data_augmentations = data_augmentations
        self.threshold_connected_components = threshold_connected_components
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.DATASET_DICT = {
            "BCSS": [self.data_fetcher.len_bcss, self.data_fetcher.get_bcss, False, 0.0],
            "CAMELYON": [self.data_fetcher.len_camelyon, self.data_fetcher.get_camelyon, True, 0.0],
            "CellSeg": [self.data_fetcher.len_cellseg, self.data_fetcher.get_cellseg, False, 0.0],
            "CoCaHis": [self.data_fetcher.len_cocahis, self.data_fetcher.get_cocahis, False, 0.0],
            "CoNIC": [self.data_fetcher.len_conic, self.data_fetcher.get_conic, False, 0.0],
            "CPM": [self.data_fetcher.len_cpm, self.data_fetcher.get_cpm, False, 0.0],
            "CRAG": [self.data_fetcher.len_crag, self.data_fetcher.get_crag, False, 1.0],
            "CryoNuSeg": [self.data_fetcher.len_cryonuseg, self.data_fetcher.get_cryonuseg, False, 0.0],
            "GlaS": [self.data_fetcher.len_glas, self.data_fetcher.get_glas, False, 1.0], 
            "ICIA2018": [self.data_fetcher.len_icia2018, self.data_fetcher.get_icia2018, True, 0.0],
            "Janowczyk": [self.data_fetcher.len_janowczyk, self.data_fetcher.get_janowczyk, False, 0.0],
            "KPI": [self.data_fetcher.len_kpi, self.data_fetcher.get_kpi, False, 0.0],
            "Kumar": [self.data_fetcher.len_kumar, self.data_fetcher.get_kumar, False, 0.0],
            "MoNuSAC": [self.data_fetcher.len_monusac, self.data_fetcher.get_monusac, False, 0.0],
            "MoNuSeg": [self.data_fetcher.len_monuseg, self.data_fetcher.get_monuseg, False, 0.0],
            "NuClick": [self.data_fetcher.len_nuclick, self.data_fetcher.get_nuclick, False, 0.0],
            "PAIP2023": [self.data_fetcher.len_paip2023, self.data_fetcher.get_paip2023, False, 0.0],
            "PanNuke": [self.data_fetcher.len_pannuke, self.data_fetcher.get_pannuke, False, 0.0],
            "SegPath": [self.data_fetcher.len_segpath, self.data_fetcher.get_segpath, False, 0.0],
            "SegPC": [self.data_fetcher.len_segpc, self.data_fetcher.get_segpc, False, 0.0],
            "TIGER": [self.data_fetcher.len_tiger, self.data_fetcher.get_tiger, False, 0.0],
            "TNBC": [self.data_fetcher.len_tnbc, self.data_fetcher.get_tnbc, False, 0.0],
        }
        self.len = 0
        for dataset in self.datasets:
            self.DATASET_DICT[dataset][0] = self.DATASET_DICT[dataset][0]()
            self.len += self.DATASET_DICT[dataset][0]
        self.resize = ResizeLongestSide(image_encoder_size)


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        for dataset in self.datasets:
            if idx < self.DATASET_DICT[dataset][0]:
                image_name, mask_name = self.DATASET_DICT[dataset][1](idx)
                return idx, DataProcessing.preprocess(
                    image_name, mask_name, self.pixel_mean, self.pixel_std, self.DATASET_DICT[dataset][2], 
                    self.image_encoder_size, self.mask_augmentation_tries, self.data_augmentations, self.threshold_connected_components
                    ), self.DATASET_DICT[dataset][3]
            else:
                idx -= self.DATASET_DICT[dataset][0]

    def get_image(self, idx):
        for dataset in self.datasets:
            if idx < self.DATASET_DICT[dataset][0]:
                image_name, mask_name = self.DATASET_DICT[dataset][1](idx)
                return image_name, mask_name
            else:
                idx -= self.DATASET_DICT[dataset][0]

def prepare_data(data_config):
    """
    
    """
    datasets = data_config["datasets"]
    data_directory = data_config["data_directory"]
    cluster = data_config["cluster"]
    image_encoder_size = data_config["image_encoder_size"]
    batch_size = data_config["batch_size"]
    drop_last = data_config["drop_last"]
    num_workers = data_config["num_workers"]
    

    # Parameters only for training
    use_holdout_testset = data_config.get("use_holdout_testset", True)
    holdout_testsets = data_config.get("test_datasets", datasets)
    train_split = data_config.get("train_split", 0.8)
    val_split = data_config.get("val_split", 0.1)
    test_split = data_config.get("test_split", 0.1)
    data_split = data_config.get("data_split", False)
    shuffle = data_config.get("shuffle", False)
    seed = data_config.get("seed", 1)
    mask_augmentation_tries = data_config.get("mask_augmentation_tries", 5)
    data_augmentations = data_config.get("data_augmentations", ["NoOp"])
    threshold_connected_components = data_config.get("threshold_connected_components", 2)

    dataset = HistologyDataset(datasets, data_directory, cluster, image_encoder_size, mask_augmentation_tries, data_augmentations, threshold_connected_components)
    if data_split:
        generator = torch.Generator().manual_seed(seed)
        if use_holdout_testset:
            train_set, val_set = random_split(dataset, [train_split + val_split, test_split],generator=generator)
            test_set = HistologyDataset(holdout_testsets, data_directory, cluster, image_encoder_size, mask_augmentation_tries, ["NoOp"], threshold_connected_components)
        else:
            train_set, val_set, test_set = random_split(dataset, [train_split, val_split, test_split],generator=generator)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers)
        return train_loader, val_loader, test_loader
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
        return dataset, dataloader
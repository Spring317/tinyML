import torch
from PIL import Image
from typing import Tuple, List, Callable, Union
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms
from ColorDistorter import ColorDistorter
from CentralCropResize import CentralCropResize
from dataset_builder.core.utility import load_manifest_parquet


class CustomDataset(Dataset):

    def __init__(
        self,
        data: Union[str, List[Tuple[str, int]]],
        train: bool = True,
        img_size: Tuple[int, int] = (50, 50),
        
    ):
        super().__init__()
        if isinstance(data, str):
            self.image_label_with_correct_labels = load_manifest_parquet(data)
        else:
            self.image_label_with_correct_labels = data
        self.train = train
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.image_label_with_correct_labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.image_label_with_correct_labels[index]
        image = Image.open(img_path).convert("RGB")

        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        color_ordering = worker_id % 4

        if self.train:
            transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    ColorDistorter(ordering=color_ordering),
                    CentralCropResize(central_fraction=0.875, size=self.img_size),
                ]
            )
        else:
            transform = transforms.Compose(
                [CentralCropResize(central_fraction=0.875, size=self.img_size)]
            )

        image = transform(image)  # type: ignore

        return image, label  # type: ignore
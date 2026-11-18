from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from ..utils.labels import CLASS_TO_IDX


class HAM10000Dataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        transform=None,
        class_to_idx=None
    ):

        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

        self.class_to_idx = (
            class_to_idx
            if class_to_idx is not None
            else CLASS_TO_IDX
        )

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_id = row["image_id"]
        label_name = row["dx"]

        image_path = (
            self.image_dir
            / f"{image_id}.jpg"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        label = self.class_to_idx[
            label_name
        ]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
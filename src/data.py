from torch.utils.data import Dataset
import torchvision.transforms.functional as transform

from PIL import Image

import os

class Dataset(Dataset):
    def __init__(self, directory, image_labels, transform = None) -> None:
        super().__init__()
        self.directory= directory
        self.labels = image_labels
        self.transform = transform

    def __len__(self):
        img_num = os.listdir(self.directory)
        return len(img_num)

    def __getitem__(self, idx):

        #https://stackoverflow.com/questions/52473516/split-dataset-based-on-file-names-in-pytorch-dataset
        image = Image.open(self.directory[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return {
            'image': transform.pil_to_tensor(image),
            'label': label
        }


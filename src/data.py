from torch.utils.data import Dataset
import torchvision.transforms.functional as transform

import PIL

import os

class Dataset(Dataset):
    def __init__(self, directory, transform = None) -> None:
        super().__init__()
        self.directory= directory
        self.augmentations = transform

    def __len__(self):
        img_num = os.listdir(self.directory)
        return len(img_num)

    def __getitem__(self, idx):
        images = [0] * self.__len__()
        labels = [0] * self.__len__()

        for i, filename in enumerate(os.listdir(self.directory)):
            images[i] = PIL.Image.open(self.directory + 
                                              filename,)
                                              
            if filename.startswith("car"):
                labels[i] = 1
            
        label = labels[idx]
        image = images[idx]

        if self.augmentations:
            image = self.augmentations(image)
        
        return {
            'image': transform.pil_to_tensor(image),
            'label': label
        }


import matplotlib.pyplot as plt
import random

class ImageVisualizer:
    def __init__(self, dataset, rows, cols, figsize = (20,10)):
        self.dataset = dataset
        self.rows = rows
        self.cols = cols
        self.figsize = figsize




    def view_random_images(self):
        num_images = self.rows * self.cols
        image_idx = random.sample(range(0, self.dataset.__len__()), num_images)

        f = plt.figure(figsize= self.figsize)

        for i, image in enumerate(image_idx):
            result = self.dataset.__getitem__(image)['image']
            f.add_subplot(self.rows, self.cols, i+1)
            plt.axis('off')
            plt.title(str(result.size()), fontsize = 15)
            plt.imshow(result.permute(1,2,0))

        plt.tight_layout()


    def view_cars():
        pass
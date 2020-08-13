import os
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CycleGANDataset(Dataset):
    def __init__(self,
                 from_dir,
                 to_dir,
                 im_size=[256, 256, 3],
                 ):

        self.from_dir = from_dir
        self.to_dir = to_dir
        self.im_size = im_size

        self.from_paths = self.process_img_dir(self.from_dir)
        self.to_paths = self.process_img_dir(self.to_dir)
        self.dataset_len = max(len(self.from_paths), len(self.to_paths))

        self.set_default_transform()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        from_path = self.from_paths[item % len(self.from_paths)]
        from_img = self.image_transforms(Image.open(from_path))

        # randomize to image for more robust cycle consistency
        to_idx = np.random.randint(0, len(self.to_dir))
        to_path = self.to_paths[to_idx]
        to_img = self.image_transforms(Image.open(to_path))


        from_ = {
            'from_img': from_img,
            'from_path': from_path
        }

        to_ = {
            'to_img': to_img,
            'to_path': to_path
        }

        return from_, to_

    def set_default_transform(self):
        t_list = [transforms.Resize(self.im_size[0]),
                   transforms.ToTensor(),
                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        self.image_transforms = transforms.Compose(t_list)

    def process_img_dir(self, dir):
        img_paths = []
        for path in os.listdir(dir):
            if path.split('.')[-1] == 'jpg':
                pth = os.path.join(dir, path)
                img_paths.append(pth)

        return img_paths
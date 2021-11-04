import torchvision.transforms as transforms
import torch.utils.data as data
import os
from PIL import Image


class GraspingDataset(data.Dataset):
    def __init__(self, data_dir):
        super(GraspingDataset, self).__init__()

        self.img_path = os.path.join(data_dir, "images")
        self.lbl_path = os.path.join(data_dir, "labels")

        self.image_filenames = [img for img in os.listdir(self.img_path)]

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):

        image = Image.open(os.path.join(self.img_path, self.image_filenames[index]))
        label = Image.open(os.path.join(self.lbl_path, self.image_filenames[index]))

        image = transforms.ToTensor()(image)
        label = transforms.ToTensor()(label)

        image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
        label = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(label)

        return image, label

    def __len__(self):
        return len(self.image_filenames)

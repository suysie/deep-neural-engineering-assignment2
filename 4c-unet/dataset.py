from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms


class DotToCharDataset(Dataset):
    def __init__(self, dots_dir, full_dir, transform=None):
        self.dots_dir = dots_dir
        self.full_dir = full_dir
        self.transform = transform
        self.files = sorted(os.listdir(dots_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        dot_img_name = self.files[idx]
        full_img_name = dot_img_name  # Assuming filenames match exactly

        dot_path = os.path.join(self.dots_dir, dot_img_name)
        full_path = os.path.join(self.full_dir, full_img_name)

        #to be parametrized later
#        dot_img = Image.open(dot_path).convert('RGB')
#        full_img = Image.open(full_path).convert('RGB')
        dot_img = Image.open(dot_path).convert('L')  # 'L' for Grayscale
        full_img = Image.open(full_path).convert('L')

        if self.transform:
            dot_img = self.transform(dot_img)
            full_img = self.transform(full_img)

        return dot_img, full_img


transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resized to a fixed size (if needed - current training is fixed at this size, to be parametrzed)
    transforms.ToTensor(),
])
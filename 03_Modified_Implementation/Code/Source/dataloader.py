import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class PairedDataset(Dataset):
    def __init__(self, low_dir, high_dir, transform=None, crop_size=None, training=True):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.transform = transform
        self.crop_size = crop_size
        self.training = training

        valid_exts = ('.png', '.jpg', '.jpeg')
        self.low_images = sorted([f for f in os.listdir(low_dir) if f.lower().endswith(valid_exts)])
        self.high_images = sorted([f for f in os.listdir(high_dir) if f.lower().endswith(valid_exts)])

        assert len(self.low_images) == len(self.high_images), f"Mismatch: {len(self.low_images)} low vs {len(self.high_images)} high"

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_path = os.path.join(self.low_dir, self.low_images[idx])
        high_path = os.path.join(self.high_dir, self.high_images[idx])

        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')

        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)

        if self.training and self.crop_size:
            i, j, h, w = transforms.RandomCrop.get_params(low_img, output_size=(self.crop_size, self.crop_size))
            low_img = transforms.functional.crop(low_img, i, j, h, w)
            high_img = transforms.functional.crop(high_img, i, j, h, w)

        return low_img, high_img

def create_dataloaders(train_low, train_high, test_low, test_high, crop_size=256, batch_size=4, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_loader, test_loader = None, None

    if train_low and train_high and os.path.exists(train_low) and os.path.exists(train_high):
        train_dataset = PairedDataset(train_low, train_high, transform=transform, crop_size=crop_size, training=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if test_low and test_high and os.path.exists(test_low) and os.path.exists(test_high):
        test_dataset = PairedDataset(test_low, test_high, transform=transform, training=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

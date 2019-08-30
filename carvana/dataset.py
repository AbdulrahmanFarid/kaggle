import os
import cv2
from PIL import Image
class carvana_dataset(Dataset):
    def __init__(self, path, train = 'train', mask = 'train_masks', transform = None):
        self.images = sorted([os.path.join(path, train, image) for image in os.listdir(os.path.join(path, train))])
        self.masks= sorted([os.path.join(path, mask, masking) for masking in os.listdir(os.path.join(path, mask))])
        self.transform = transform
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        mask = np.asarray(Image.open(self.masks[idx]))

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return {'image': image, 'mask': mask}

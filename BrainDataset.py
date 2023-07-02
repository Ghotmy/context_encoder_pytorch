from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import DataLoader


class BrainDataset(Dataset):

    def __init__(self, axis, length):
        self.axis = axis
        self.length = length
        if axis == "Sagital":
            self.prefix = "sag"
        elif axis == "Coronal":
            self.prefix = "cor"

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        patient = idx // 215
        i = idx % 215
        img = read_image(f'/home/ghotmy/College/graduation/Brain_MRI_inpainting_GAN/Training_Data/{self.axis}/{self.prefix}_{patient}_{i}.png', mode=ImageReadMode.GRAY)
        img_masked = read_image(f'/home/ghotmy/College/graduation/Brain_MRI_inpainting_GAN/Training_Data/{self.axis}/{self.prefix}_{patient}_{i}_msk.png', mode=ImageReadMode.GRAY)

        return img, img_masked


if __name__ == '__main__':
    BATCH_SIZE = 128

    sagital_data = BrainDataset("Sagital", 215 * 2)
    sagital_dataloader = DataLoader(sagital_data, batch_size=BATCH_SIZE)

    coronal_data = BrainDataset("Coronal", 215 * 2)
    coronal_dataloader = DataLoader(coronal_data, batch_size=BATCH_SIZE)

    for i1, i2, i3, i4 in coronal_dataloader:
        print(i1.shape)
        plt.figure(figsize=(15, 5))
        plt.style.use('grayscale')
        plt.subplot(141)
        plt.imshow(i1[1][0])
        plt.subplot(142)
        plt.imshow(i2[1][0])
        plt.subplot(143)
        plt.imshow(i3[1][0])
        plt.subplot(144)
        plt.imshow(i4[1][0])
        plt.show()

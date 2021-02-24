import torch
import torchvision
import numpy as np


if __name__ == '__main__':
    dataset = torchvision.datasets.ImageFolder(
            root=r'###',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.ToTensor()
                
    ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)

    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    
    for i, (image, _) in enumerate(dataloader, 0):
        numpy_image = image.numpy()
        
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std0 = np.std(numpy_image, axis=(0,2,3))
        batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
        
        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)
        
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    pop_std1 = np.array(pop_std1).mean(axis=0)

    print(pop_mean, pop_std0, pop_std1)
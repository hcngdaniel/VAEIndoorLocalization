#!/usr/bin/env python3
from torchvision import transforms
import torch
import numpy as np
import cv2


from models.model1 import Model
from dataset import Dataset


dataset = Dataset(
    dataset_name='vision_slam',
)

model = Model()
model.to('cpu')
model.load_state_dict(torch.load('saves/model1-exp5.pth', map_location='cpu'))

encoder = model.encoder
decoder = model.decoder
transformer = model.transformer
print('initialized')
print(len(dataset))

while True:
    try:
        idx = int(torch.randint(0, len(dataset), [1]))
        image, (norm, theta, alpha) = dataset[idx]
        norm, theta, alpha = float(norm), float(theta), float(alpha)
        cv2_img = (image * 127.5) + 127.5
        cv2_img = np.array(transforms.ToPILImage()(cv2_img), dtype='uint8')
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("img", cv2_img)
        print(f'real: {norm} {theta} {alpha}')
        mu, logvar = encoder(torch.unsqueeze(image, 0))
        fake_norm, fake_theta, fake_alpha = map(float, transformer(mu))
        print(f'fake: {fake_norm} {fake_theta} {fake_alpha}')
        cv2.waitKey(-1)

    except KeyboardInterrupt:
        exit()

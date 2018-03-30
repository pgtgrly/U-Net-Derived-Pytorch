import random
import cv2


def unet_augment(sample,vertical_prob,horizontal_prob):
    image, masks= sample['image'], sample['masks']

    if (random.random()<vertical_prob):
        image=cv2.flip(image,1)
        masks=cv2.flip(masks,1)


    if (random.random()<horizontal_prob):
        image=cv2.flip(image,0)
        masks=cv2.flip(masks,0)


    return {'image': image, 'masks': masks}

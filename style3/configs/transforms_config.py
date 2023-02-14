import torch
import torchvision.transforms as transforms
from abc import abstractmethod


class TransformsConfig:

    def __init__(self, opts):
        self.opts = opts

    @abstractmethod
    def get_transforms(self):
        pass


class EncodeTransforms(TransformsConfig):

    def __init__(self, opts):
        super(EncodeTransforms, self).__init__(opts)

    def get_transforms(self):
        transforms_dict = {
            'transform_gt_train': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_source': None,
            'transform_test': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_inference': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }
        return transforms_dict


class MedTransforms(TransformsConfig):

    def __init__(self, opts):
        super(MedTransforms, self).__init__(opts)
        img_chn = 6 if 'rxrx19b' in self.opts.dataset_type else 3
        self.mean = [0.5] * img_chn
        self.std = [0.5] * img_chn

    def get_transforms(self):
        angles = [0, 90, 180, 270]

        def random_rotation(x):
            angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
            if angle > 0:
                x = transforms.functional.rotate(x, angle)
            return x
        t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

        transforms_dict = {
            'transform_gt_train': transforms.Compose([
                transforms.ToTensor(),
                t_random_rotation,
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(self.mean, self.std)]),
            'transform_source': None,
            'transform_test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)]),
            'transform_inference': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)]),
        }
        return transforms_dict

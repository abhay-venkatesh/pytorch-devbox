# Reference: https://github.com/pytorch/vision/blob/master/torchvision/datasets/coco.py

from PIL import Image, ImageDraw
from skimage.transform import resize
import csv
import numpy as np
import os
import os.path
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class CocoBbox(data.Dataset):
    def __init__(self, root, ann_file_path, transform=None):
        self.root = root
        self.img_names = []
        self.transform = transform
        with open(ann_file_path, newline='') as ann_file:
            reader = csv.reader(ann_file, delimiter=',')
            for row in reader:
                self.img_names.append(row[0])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_name = self.img_names[index]
        img_path = self.root + "/images/" + img_name
        img = Image.open(img_path).convert('RGB')
        seg_name = img_name.replace(".jpg", ".png")
        seg_path = self.root + "/annotations/" + seg_name
        seg = Image.open(seg_path)
        bbox_path = self.root + "/bbox/" + seg_name
        bbox = Image.open(bbox_path)

        img = self.transform(img)
        seg = self.transform(seg)
        bbox = self.transform(bbox)
        return img, (seg, bbox)

    def __len__(self):
        return len(self.img_names)


class CocoStuff(data.Dataset):
    """ Binary stuff classification reader """

    def __init__(self,
                 root,
                 annFile,
                 target_class=157,
                 transform=None,
                 target_transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.target_class = target_class
        self.total_steps = len(self.coco.getImgIds(catIds=target_class))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        target, contains_class = self._anns_to_seg(img_id, anns,
                                                   self.target_class)
        target_ = resize(
            target, (426, 640), anti_aliasing=False, mode='constant')
        target_ = np.where(target_ > 0, 1, 0)
        target = torch.tensor(target_, dtype=torch.float32)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, (target, contains_class)

    def _print_np_array_mask(self, mask):
        mask_ = np.multiply(mask, 200)
        Image.fromarray(mask_).show()

    def _get_cat_from_ann(self, ann):
        cats = self.coco.loadCats(ids=ann["category_id"])
        print(cats)

    def _draw_bbox_mask(self, img_id, bbox):
        img_height = self.coco.loadImgs(img_id)[0]['height']
        img_width = self.coco.loadImgs(img_id)[0]['width']
        seg = Image.fromarray(np.zeros((img_height, img_width)))
        draw = ImageDraw.Draw(seg)
        x, y, mask_width, mask_height = bbox
        rect = self._get_rect(x, y, mask_width, mask_height, 0)
        draw.polygon([tuple(p) for p in rect], fill=1)
        np_seg = np.asarray(seg)
        return np_seg

    def _get_rect(self, x, y, width, height, angle):
        # Reference: https://stackoverflow.com/questions/12638790/drawing-a-rectangle-inside-a-2d-numpy-array
        rect = np.array([(0, 0), (width, 0), (width, height), (0, height),
                         (0, 0)])
        theta = (np.pi / 180.0) * angle
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        offset = np.array([x, y])
        transformed_rect = np.dot(rect, R) + offset
        return transformed_rect

    def _anns_to_seg(self,
                     img_id,
                     anns,
                     positive_class=157,
                     height=426,
                     width=640):

        for ann in anns:
            if ann["category_id"] == positive_class:
                bbox = ann["bbox"]
                seg = self._draw_bbox_mask(img_id, bbox)
                return seg, 1
        arr = np.zeros((height, width))
        return arr, 0

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp,
            self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp,
            self.target_transform.__repr__().replace('\n',
                                                     '\n' + ' ' * len(tmp)))
        return fmt_str


class CocoCaptions(data.Dataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#captions-challenge2015>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    Example:
        .. code:: python
            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root = 'dir where images are',
                                    annFile = 'json annotation file',
                                    transform=transforms.ToTensor())
            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample
            print("Image Size: ", img.size())
            print(target)
        Output: ::
            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']
    """

    def __init__(self, root, annFile, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        target = [ann['caption'] for ann in anns]

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)


class CocoDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp,
            self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp,
            self.target_transform.__repr__().replace('\n',
                                                     '\n' + ' ' * len(tmp)))
        return fmt_str
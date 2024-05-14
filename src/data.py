import os
import pandas as pd
import numpy as np
import itertools
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import cv2
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from tqdm import tqdm
import warnings
from PIL import Image
warnings.simplefilter(action='ignore')

from .utils.data_utils import get_scan_file_path, decode_rle, convert_binary_mask_to_rle
# from ..config import BASE_PATH, CLASSES


BASE_PATH = '../../gi-tract-image-segmentation'
CLASSES = ['small_bowel', 'large_bowel', 'stomach']
IMAGE_SIZE = (128, 128)


class SegmentationDataset():
    def __init__(self, dataset_dir, csv_file_path):
        self.dataset_dir = dataset_dir
        self.train_csv = pd.read_csv(csv_file_path)
        self.processed_df = self.preprocess(self.train_csv)
        # self.categories = self.create_coco_categories(CLASSES)
        # self.images = self.create_coco_images(self.processed_df)
        # self.annotations = self.create_annotations(self.processed_df, self.images)


    def preprocess(self, df):

        df['case'] = df['id'].apply(lambda id_str: id_str.split('_')[0][4:])
        df['day'] = df['id'].apply(lambda id_str: id_str.split('_')[1][3:])
        df['slice'] = df['id'].apply(lambda id_str: id_str.split('_')[-1])
        df['file_path'] = df['id'].apply(lambda id_str: get_scan_file_path(self.dataset_dir, id_str))

        df['file_name'] = df['file_path'].apply(lambda path: os.path.basename(path))
        df['composite_id'] = df.apply(lambda row: f"{row['case']}_{row['day']}_{row['file_name']}", axis=1)

        df['image_height'] = df['file_name'].apply(lambda name: int(name.split('_')[2]))
        df['image_width'] = df['file_name'].apply(lambda name: int(name.split('_')[3]))
        df['resolution'] = df.apply(lambda row: f"{row['image_height']}x{row['image_width']}", axis=1)

        masked_df = df[df['segmentation'].notnull()]
        masked_df["segmentation"] = masked_df["segmentation"].astype("str")
        masked_df = masked_df.reset_index(drop=True)

        return masked_df

    def create_coco_categories(self, classes):
        """ Create categories section for COCO JSON. """
        categories = [{"id": idx, "name": cls} for idx, cls in enumerate(classes)]
        return categories

    def create_coco_images(self, df):
        images = []
        filepaths = df.file_path.unique().tolist()

        for i, filepath in enumerate(tqdm(filepaths, desc="Processing images")):
            file_name = '/'.join(filepath.split("/")[2:])
            height = int(filepath.split("/")[-1].split("_")[3])
            width = int(filepath.split("/")[-1].split("_")[2])
            images.append({
                "id": i + 1,
                "file_name": file_name,
                "width": width,
                "height": height
            })
        return images

    def create_annotations(self, df, images):
        annotations = []
        count = 0 

        for image in tqdm(images, desc='Generating annotations'):
            image_id = image['id']
            filepath = image['file_name']
            file_id = ('_'.join(
                (filepath.split("/")[-3] + "_" + filepath.split("/")[-1]).split("_")[:-4]))
            height_slice = int(filepath.split("/")[-1].split("_")[3])
            width_slice = int(filepath.split("/")[-1].split("_")[2])
        
            ids = df.index[df['id'] == file_id].tolist()
            if len(ids) > 0:
                for idx in ids:
                    segmentation_mask = decode_rle(
                        df.iloc[idx]['segmentation'], (height_slice, width_slice))
                    for contour in cv2.findContours(segmentation_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]:
                        mask_image = np.zeros(segmentation_mask.shape, dtype=np.uint8)
                        cv2.drawContours(mask_image, [contour], -1, 255, -1)
                        encoded_segmentation = convert_binary_mask_to_rle(mask_image)
                        ys, xs = np.where(mask_image)
                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                        annotations.append({
                            'segmentation': encoded_segmentation,
                            'bbox': [x1, y1, x2 - x1 + 1, y2 - y1 + 1],  # (x, y, w, h) format
                            'area': mask_image.sum(),
                            'image_id': image_id,
                            'category_id': CLASSES.index(df.iloc[idx]['class']),
                            'iscrowd': 0,
                            'id': count
                        })
                        count += 1
        return annotations

class RandomTransforms:
    """ This class applies the same random transformations to both images and masks. """
    def __init__(self):
        self.color_jitter = transforms.ColorJitter(brightness=0.05, contrast=0.05)

    def __call__(self, image, masks):
        
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image.astype('uint8'), 'RGB')
        if not isinstance(masks[0], Image.Image):
            masks = [Image.fromarray(mask.astype('uint8'), 'L') for mask in masks]  # Mask should be converted to 'L' mode

        if random.random() > 0.5:
            image = self.color_jitter(image)

        # Apply random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            masks = [TF.hflip(mask) for mask in masks]     

        if random.random() > 0.5:
            angle = random.randint(-5, 5)  # Rotation degree
            image = TF.rotate(image, angle)
            masks = [TF.rotate(mask, angle) for mask in masks]

        image = np.array(image)[:, :, ::-1]
        masks = [np.array(mask) for mask in masks]

        return image, masks
    
class DataGenerator(Dataset):
    def __init__(self, dataset_dir, subset, classes, 
                 input_image_size, annFile, shuffle=False, transform=None):
        self.dataset_dir = dataset_dir
        self.subset = subset
        self.classes= classes
        self.coco = COCO(annFile)
        self.catIds = self.coco.getCatIds(catNms=self.classes)
        self.cats = self.coco.loadCats(self.catIds)
        self.imgIds = self.coco.getImgIds()
        self.image_list = self.coco.loadImgs(self.imgIds)
        self.indexes = np.arange(len(self.image_list))
        self.input_image_size = (input_image_size)
        self.dataset_size = len(self.image_list)
        self.transform = RandomTransforms() if transform else None
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
      return int(len(self.image_list))

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)      

    def get_class_name(self, class_id, cats):
        for i in range(len(cats)):
            if cats[i]['id'] == class_id:
                return cats[i]['name']
        return None
  
    def get_normal_mask(self, image_id, catIds):
        annIds = self.coco.getAnnIds(image_id, catIds=catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        cats = self.coco.loadCats(catIds)
        train_mask = np.zeros(self.input_image_size, dtype=np.uint8)
        for a in range(len(anns)):
            className = self.get_class_name(anns[a]['category_id'], cats)
            pixel_value = self.classes.index(className)+1
            new_mask = cv2.resize(self.coco.annToMask(anns[a])*pixel_value, self.input_image_size)
            train_mask = np.maximum(new_mask, train_mask)
        return train_mask          
        

    def get_levels_mask(self, image_id):
      #for each category , we get the x mask and add it to mask list
      masks = []  
      mask = np.zeros((self.input_image_size))
      for catId in self.catIds:
        mask = self.get_normal_mask(image_id, catId)
        masks.append(mask)
      return masks       

    def get_image(self, file_path):
        full_path = os.path.join(self.dataset_dir, file_path)
        train_img = cv2.imread(full_path, cv2.IMREAD_ANYDEPTH)
        if train_img is None:
            raise ValueError(f"Unable to load image from path: {full_path}")
        train_img = cv2.resize(train_img, (self.input_image_size))
        train_img = train_img.astype(np.float32) / 255.
        if (len(train_img.shape)==3 and train_img.shape[2]==3): 
            return train_img
        else: 
            stacked_img = np.stack((train_img,)*3, axis=-1)
            return stacked_img          
    
    def __getitem__(self, index):
        
        X = np.empty((128, 128, 3))
        y = np.empty((128, 128, 3))

        img_info = self.image_list[index]

        X = self.get_image(img_info['file_name'])
        mask_train = self.get_levels_mask(img_info['id'])

        if self.transform:
            X, mask_train = self.transform(X, mask_train)

        for j in self.catIds:
            y[:, :, j] = mask_train[j]

        X = np.array(X)
        y = np.array(y)

        if self.subset == 'train':
            return X, y
        else: 
            return X
import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader

from .oxford_pets import OxfordPets
from .imagenet import ImageNet


template = ['a photo of a {}.']


class SUN397(DatasetBase):

    dataset_dir = 'SUN397'

    def __init__(self, cfg, clean_data=False):

        root = cfg['root_path']
        num_shots = cfg['shots']
        open_world = cfg['open_world']
        num_classes = cfg['num_classes']

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        if clean_data == False:
            self.split_path = os.path.join(self.dataset_dir, 'split_sun397.json')
            train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
            train = self.generate_fewshot_dataset(train, num_shots=num_shots) 
        else:        
            self.split_path = os.path.join(self.dataset_dir, 'split_cleaned_SUN397.json')
            train = ImageNet.read_split_(self.split_path, self.dataset_dir)
            val = None
            test = None  
        
        self.template = template
        super().__init__(train_x=train, val=val, test=test, open_world=open_world, num_classes=num_classes, clean_data=clean_data)
    


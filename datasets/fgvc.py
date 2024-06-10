import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from .imagenet import ImageNet
import random



template = ['a photo of a {}, a type of aircraft.']


class FGVCAircraft(DatasetBase):

    dataset_dir = 'fgvc_aircraft/data'

    def __init__(self, cfg, clean_data=False):

        root = cfg['root_path']
        num_shots = cfg['shots']
        open_world = cfg['open_world']
        num_classes = cfg['num_classes']


        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')

        self.template = template
        if clean_data == False:
            classnames = []
            with open(os.path.join(self.dataset_dir, 'variants.txt'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    classnames.append(line.strip())
            cname2lab = {c: i for i, c in enumerate(classnames)}

            train = self.read_data(cname2lab, 'images_variant_train.txt')
            val = self.read_data(cname2lab, 'images_variant_val.txt')
            test = self.read_data(cname2lab, 'images_variant_test.txt')
            
            train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        else:
            self.split_path = os.path.join(self.dataset_dir, 'split_cleaned_fgvc.json')
            train = ImageNet.read_split_(self.split_path, self.image_dir)
            val = None
            test = None  

        super().__init__(train_x=train, val=val, test=test, open_world=open_world, num_classes=num_classes, clean_data=clean_data)
       
    
    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                imname = line[0] + '.jpg'
                classname = ' '.join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                incorrect_label = random.choice([i for i in range(0, 99) if i != label]) # 随机生成错误类别
                incorrect_classname = next(key for key, value in cname2lab.items() if value == incorrect_label)
                
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=classname,
                    incorrect_label = int(incorrect_label),
                    incorrect_classname = incorrect_classname
                )
                items.append(item)
        
        return items
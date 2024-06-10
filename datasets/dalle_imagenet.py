import os
from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
# from .imagenet_ import ImageNet

class Dalle_Imagenet(DatasetBase):
    
    dataset_dir = 'dalle_imagenet'
    dataset_dir_ = 'dalle_imagenet_multistyle'

    def __init__(self, cfg, clean_data=False):
        root = cfg['root_path']
        num_shots = cfg['dalle_shots']
        open_world = cfg['open_world']
        num_classes = cfg['num_classes']

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.split_path = os.path.join(self.dataset_dir, 'split_dalle_imagenet.json')
        train, val = self.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        # if cfg['open_world']['is_open_world'] == True and cfg['open_world']['is_open_world'] != 0.0:
        #     self.dataset_dir_ = os.path.join(root, self.dataset_dir_)
        #     self.image_dir_ = os.path.join(self.dataset_dir_, 'images')
        #     self.split_path_ = os.path.join(self.dataset_dir_, 'split_dalle_imagenet_multistyle.json')
        #     train_, val_ = self.read_split(self.split_path_, self.image_dir_)  
        #     train_ = self.generate_fewshot_dataset(train_, num_shots=num_shots)      
        #     train.extend(train_)
        
        super().__init__(train_x=train, val=val, open_world=open_world, num_classes=num_classes)
    
    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname, domain in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(
                    impath = impath,
                    label = int(label),
                    classname = classname,
                    domain = domain
                )
                out.append(item)
            return out
        
        print(f'Reading split from {filepath}')
        split = read_json(filepath)
        train = _convert(split['train'])
        val = _convert(split['val'])
        # test = _convert(split['test'])

        return train, val
import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
import gdown
import json
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as T
from PIL import Image
import math

def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    if not osp.exists(path):
        raise IOError('No file exists at {}'.format(path))

    while True:
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print(
                'Cannot read image from {}, '
                'probably due to heavy IO. Will re-try'.format(path)
            )


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith('.') and 'sh' not in f]
    if sort:
        items.sort()
    return items


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        classname (str): class name.
        incorrect_label (int): incorrect class label.
        incorrect_classname (str): incorrect class name.
        domain (int): domain label.
    """

    def __init__(self, impath='', label=0, classname='', incorrect_label=0, incorrect_classname='', domain=-1):
        assert isinstance(impath, str)
        assert isinstance(label, int)
        assert isinstance(classname, str)
        assert isinstance(incorrect_label, int)
        assert isinstance(incorrect_classname, str)
        assert isinstance(domain, int)

        self._impath = impath
        self._label = label
        self._classname = classname
        self._incorrect_label = incorrect_label
        self._incorrect_classname = incorrect_classname
        self._domain = domain

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def classname(self):
        return self._classname

    @property
    def incorrect_label(self):
        return self._incorrect_label

    @property
    def incorrect_classname(self):
        return self._incorrect_classname
    
    @property
    def domain(self):
        return self._domain

class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """
    dataset_dir = '' # the directory where the dataset is stored
    domains = [] # string names of all domains

    def __init__(self, train_x=None, train_u=None, val=None, test=None, open_world=None, num_classes=None, clean_data=False, open_set=False):
        self._train_x = train_x # labeled training data
        self._train_u = train_u # unlabeled training data (optional)
        self._val = val # validation data (optional)
        self._test = test # test data
        self._open_world = open_world
        self._num_classes = num_classes

        self._lab2cname, self._classnames = self.get_lab2cname(train_x)
        if clean_data == True: # not open-world setting
            self._noise_img_ids, self._noise_data, self._nciai = None, None, None # nciai: noise_class_ids_all_incorrect
        else: # is open-world setting, giving wrong labels for patial training data
            if open_world['is_open_world'] == False:
                self._noise_img_ids, self._noise_data, self._nciai = None, None, None
            elif open_world['is_open_world'] == True and open_world['nlb'] == 0.0:
                self._noise_img_ids, self._noise_data, self._nciai = None, None, None
            else:
                # construct imgs with noise labels
                self._noise_img_ids, self._noise_data, self._noise_class_ids, self._nciai = self.construct_noise_data(train_x) # nciai: noise_class_ids_all_incorrect

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def noise_img_ids(self):
        return self._noise_img_ids

    @property
    def noise_data(self):
        return self._noise_data

    @property
    def noise_class_ids_all_incorrect(self):
        return self._nciai    

    @property
    def noise_class_ids(self):
        return self._noise_class_ids    
    


    def construct_noise_data(self, data_source):
        """Construct noise data.

        Args:
            data_source (list): a list of Datum objects.
        """        
        
        noise_num = math.ceil(len(data_source) * self._open_world['nlb']) # the num of noisy data

        # Count how many samples there are in each class and display the samples corresponding to each class.
        dict_tmp = {}
        for ii in range(len(data_source)):
            lb_tmp = data_source[ii].label
            if lb_tmp in dict_tmp:
                dict_tmp[lb_tmp].append(ii)
            else:
                dict_tmp[lb_tmp] = [ii]

        # Randomly select 100 categories and let all the data corresponding to these categores be used as noise data; the remaining noise data are randomly selected
        shot_num = len(dict_tmp[0])
        random_classes = int(self._num_classes / 10)
        noise_class_ids_all_incorrect = random.sample(range(len(dict_tmp)), random_classes) # All data in this class are wrong labels
        noise_img_ids = [dict_tmp[jj][kk] for jj in noise_class_ids_all_incorrect for kk in range(shot_num)]
        other_noise_img_ids = random.sample([num for num in range(len(data_source)) if num not in noise_img_ids], noise_num-len(noise_img_ids))
        noise_img_ids.extend(other_noise_img_ids)

        # Count the categories of data with wrong labels
        noise_class_ids = []
        for ii in noise_img_ids:
            lb_tmp = data_source[ii].label
            if lb_tmp in noise_class_ids:
                continue
            else:
                noise_class_ids.append(lb_tmp)

        noise_data = {}
        count = 0
        for item in data_source:
            if count in noise_img_ids:
                noise_data[count] = [item]
            count = count + 1

        return noise_img_ids, noise_data, noise_class_ids, noise_class_ids_all_incorrect

    
    def get_num_classes(self, data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_num_classes_(self, data_source, noise_img_ids):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        count = 0
        for item in data_source:
            if count in noise_img_ids:
                label_set.add(item.incorrect_label)
            else:
                label_set.add(item.label)
            count = count + 1
        return len(label_set)

    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        # if 'open class' not in classnames:
        #     classnames.append('open class')
        return mapping, classnames

    # def get_lab2cname_(self, data_source, noise_img_ids):
    #     """Get a label-to-classname mapping (dict).

    #     Args:
    #         data_source (list): a list of Datum objects.
    #     """
    #     container = set()
    #     count = 0
    #     for item in data_source:
    #         if count in noise_img_ids:
    #             container.add((item.incorrect_label, item.incorrect_classname))
    #         else:
    #             container.add((item.label, item.classname))
    #         count = count + 1
    #     mapping = {label: classname for label, classname in container}
    #     labels = list(mapping.keys())
    #     labels.sort()
    #     classnames = [mapping[label] for label in labels]
    #     return mapping, classnames

    def check_input_domains(self, source_domains, target_domains):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    'Input domain must belong to {}, '
                    'but got [{}]'.format(self.domains, domain)
                )

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print('Extracting file ...')

        try:
            tar = tarfile.open(dst)
            tar.extractall(path=osp.dirname(dst))
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(dst, 'r')
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        print('File extracted to {}'.format(osp.dirname(dst)))

    def generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=True
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a few number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed.
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f'Creating a {num_shots}-shot dataset')

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output


    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

    def split_dataset_by_domain(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into domain-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.domain].append(item)

        return output



class DatasetWrapper(TorchDataset):
    def __init__(self, cfg, data_source, noise_data, noise_class, input_size, transform=None, is_train=False, img_type='original',
                 return_img0=False, k_tfm=1):
        self.cfg = cfg
        self.data_source = data_source
        self.noise_data = noise_data
        self.noise_class = noise_class # all the data has wrong labels in these classes
        self.transform = transform # accept list (tuple) as input
        self.is_train = is_train
        self.img_type = img_type
        
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = k_tfm if is_train else 1
        self.return_img0 = return_img0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                'Cannot augment the image {} times '
                'because transform is None'.format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = T.InterpolationMode.BICUBIC
        to_tensor = []
        to_tensor += [T.Resize(input_size, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        normalize = T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        )
        to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]


        if self.cfg['open_world']['is_open_world'] == True and self.is_train == True: # is training data & open-world data
            
            if self.img_type == 'original':  # original training data
                if self.noise_data != None and idx in self.noise_data.keys(): # noisy data
                    output = {
                        'label': item.incorrect_label,
                        'classname': item.incorrect_classname,
                        'gt_label': item.label,
                        'gt_classname': item.classname,
                        'impath': item.impath,
                    }
                else:  # not noisy data
                    output = {
                        'label': item.label,
                        'classname': item.classname,
                        'gt_label': item.label,
                        'gt_classname': item.classname,                        
                        'impath': item.impath,
                    }  
            elif self.img_type == 'cleaned':  # corrected training data
                    output = {
                        'label': item.incorrect_label,
                        'classname': item.incorrect_classname,
                        'gt_label': item.label,
                        'gt_classname': item.classname,                          
                        'impath': item.impath,
                    }  
            elif self.img_type == 'dalle':   # training data based DALLE  
                    output = {
                        'label': item.label,
                        'classname': item.classname,
                        'gt_label': item.label,
                        'gt_classname': item.classname,                          
                        'impath': item.impath,
                    }
            # elif self.img_type == 'open':  # open set  
            #         output = {
            #             'label': item.label,
            #             'classname': item.classname,
            #             'gt_label': item.label,
            #             'gt_classname': item.classname,                          
            #             'impath': item.impath,
            #         }                          
                                
        else: # not training data
            output = {
                'label': item.label,
                'classname': item.classname,
                'gt_label': item.label,
                'gt_classname': item.classname,                  
                'impath': item.impath,
            }
     

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = 'img'
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output['img'] = img

        if self.return_img0:
            output['img0'] = self.to_tensor(img0)

        
        return output['img'], output['label'], output['gt_label'], output['impath']

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


def build_data_loader(
    cfg,
    data_source=None,
    noise_data=None,
    noise_class=None,
    img_type='original',
    batch_size=64,
    input_size=224,
    tfm=None,
    is_train=True,
    shuffle=False,
    dataset_wrapper=None
):

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, noise_data, noise_class, input_size=input_size, transform=tfm, is_train=is_train, img_type=img_type),
        batch_size=batch_size,
        num_workers=8,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=False
    )
    assert len(data_loader) > 0

    return data_loader

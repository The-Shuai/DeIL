import os
import random
import argparse
import yaml
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models as torchvision_models

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
import itertools
import json
from clean_dataset import clean_model
from models.adapters import Adapter
import open_clip
from open_clip.transform import image_transform

# 'imagenet', 'pets', 'caltech101', 'dtd', 'food101', 'sun', 'cars', 'ucf', 'eurosat', 'fgvc', 'oxford_flower' 
main_path = './configs/imagenet/config.yaml'

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings in yaml format', type=str, default=main_path)
    args = parser.parse_args()
    return args


def run_Adapter(cfg,
                adapter,
                clip,
                clipn,
                train_loader,
                dalle_train_loader,
                test_loader,
                classname_clip_text_feas,
                train_data_dict,
                dalle_data_dict,
                test_data_dict,
                dataset
                ):

    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.95, last_epoch=-1)

    optimizer = torch.optim.AdamW(
    itertools.chain(adapter.parameters()),
    lr=cfg['lr'], 
    eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader))
    best_acc, best_epoch = 0.0, 0
    alpha, beta, gamma = cfg['alpha'], cfg['beta'], cfg['gamma']
    acc_lst, loss_lst, cls_loss_lst, loss_contrastive_lst = [], [], [], []
    for epoch in range(cfg['train_epoch']):
        adapter.train()
        correct_samples, all_samples = 0, 0
        print('Train Epoch: {:} / {:}'.format(epoch, cfg['train_epoch']))

        # origin img
        # for i, (imgs, target, ignore, classname, impath) in enumerate(tqdm(train_loader)):
        for i, (imgs, lbs, gt_lbs, impath) in enumerate(train_loader):

            batchsize = len(imgs)
            imgs, lbs, gt_lbs = imgs.cuda(), lbs.cuda(), gt_lbs.cuda()
            categories = torch.unique(lbs) 
            with torch.no_grad():
                img_feas = clip.encode_image(imgs)
                img_feas /= img_feas.norm(dim=-1, keepdim=True)
            init_clip_logits = 100. * img_feas @ classname_clip_text_feas
            adapter_feas, adapter_logits = adapter(img_feas) 
            adapter_logits = logits_fuse(init_clip_logits, adapter_logits)
            adapter_logits_ = ((-1) * (beta - beta * adapter_logits)).exp()
            cls_logits = init_clip_logits +  alpha * adapter_logits_
            cls_loss = F.cross_entropy(cls_logits, lbs)

            probs_cls = F.softmax(cls_logits, dim=1)
            pred_lbs = torch.argmax(probs_cls, dim=1)

            probs_no = train_data_dict['probs_no'][batchsize*i : batchsize*(i+1)]

            
            # contrastive learning
            train_data_dict_tmp = {
            'img_feas': adapter_feas,
            'lbs': pred_lbs,
            'logits_yes': cls_logits,
            'probs_no': probs_no
            }
            anchor_feas, anchor_lbs, positive_feas, positive_lbs, negative_feas, negative_lbs = get_contrastive_data(epoch, adapter, categories, cfg, dalle_data_dict, train_data_dict_tmp)
            contrastive_feas = torch.cat((positive_feas, negative_feas), dim=1)  # (num_queries, 1 + num_negative, num_feat)            
            contrastive_logits = torch.cosine_similarity(anchor_feas, contrastive_feas, dim=2)
            temp = 0.1
            contrastive_loss = F.cross_entropy(
                contrastive_logits / temp, torch.zeros(len(categories)*cfg['anchor_num_per_class']).long().cuda()
            )

            # loss_contrastive = SupConLoss(anchor_feas, anchor_lbs, positive_feas, positive_lbs, negative_feas, negative_lbs)

            loss = cls_loss + gamma*contrastive_loss
            # loss = cls_loss
            
            acc = cls_acc(cls_logits, lbs)
            correct_samples += acc / 100 * len(cls_logits)
            all_samples += len(cls_logits)

            # torch.autograd.set_detect_anomaly(True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        acc_lst.append(acc)
        loss_lst.append(loss.item())
        cls_loss_lst.append(cls_loss.item())
        loss_contrastive_lst.append(contrastive_loss.item())

        # dalle img
        # for i, (imgs, target, ignore, classname, impath) in enumerate(tqdm(dalle_train_loader)):
        for i, (imgs, lbs, gt_lbs, impath) in enumerate(dalle_train_loader):
            batchsize = len(imgs)
            imgs, lbs, gt_lbs = imgs.cuda(), lbs.cuda(), gt_lbs.cuda()
            categories = torch.unique(lbs) 
            with torch.no_grad():
                img_feas = clip.encode_image(imgs)
                img_feas /= img_feas.norm(dim=-1, keepdim=True)
            init_clip_logits = 100. * img_feas @ classname_clip_text_feas
            adapter_feas, adapter_logits = adapter(img_feas) 
            adapter_logits = logits_fuse(init_clip_logits, adapter_logits)
            adapter_logits_ = ((-1) * (beta - beta * adapter_logits)).exp()
            cls_logits = init_clip_logits +  alpha * adapter_logits_
            cls_loss = F.cross_entropy(cls_logits, lbs)
            loss = cls_loss

            acc = cls_acc(cls_logits, lbs)
            correct_samples += acc / 100 * len(cls_logits)
            all_samples += len(cls_logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    
        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_lst)/len(loss_lst)))

        # Eval
        # val img
        # correct_samples, all_samples = 0, 0
        adapter.eval()   
        test_img_feas = test_data_dict['img_feas'] 
        test_lbs = test_data_dict['lbs'] 
        test_impaths = test_data_dict['impaths']

        test_init_clip_logits = 100. * test_img_feas @ classname_clip_text_feas
        with torch.no_grad():
            test_adapter_feas, test_adapter_logits = adapter(test_img_feas) 
        
        # test_adapter_logits = logits_fuse(test_init_clip_logits, test_adapter_logits)
        test_adapter_logits_ = ((-1) * (beta - beta * test_adapter_logits)).exp()
        test_cls_logits = test_init_clip_logits +  alpha * test_adapter_logits_
        acc = cls_acc(test_cls_logits, test_lbs)
        print('test_Acc: {:.4f}'.format(acc))

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            save_dict = {
                        'model': adapter.state_dict()
                        # 'optimizer': optimizer.state_dict()
                        }
            torch.save(save_dict,
                       os.path.join(cfg['results_dir'] + "/best_adapter" + str(cfg['shots']) + 'shots_' + str(cfg['open_world']['nlb']) + 'noise_lb.pt')
            )
    print(f"**** After fine-tuning, best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")


    print("\n-------- Searching hyperparameters on the val set. --------")
    # Search Hyperparameters
    save_path = os.path.join(cfg['results_dir'] + "/best_adapter" + str(cfg['shots']) + 'shots_' + str(cfg['open_world']['nlb']) + 'noise_lb.pt')
    state_dict = torch.load(save_path)
    state_dict_model = state_dict['model']
    adapter.load_state_dict(state_dict_model, strict=True)  
    acc = search_hp(cfg, adapter, test_img_feas, test_lbs, classname_clip_text_feas)
    if acc > best_acc:
        best_acc = acc
    
    print(f"**** After searching hyperparameters, best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    
    # print(acc_lst)
    # print(cls_loss_lst)
    # print(loss_contrastive_lst)



def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    results_dir = os.path.join('./results', cfg['dataset'], cfg['clip_backbone'])
    os.makedirs(results_dir, exist_ok=True)
    cfg['results_dir'] = results_dir
    
    cache_dir = os.path.join('./caches', cfg['dataset'], cfg['clip_backbone'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    clipn_cache_dir = os.path.join('./clipn_caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['clipn_cache_dir'] = clipn_cache_dir


    print("\nRunning configs.")
    print(cfg, "\n")

    # load frozen CLIP model
    clip_model, preprocess = clip.load(cfg['clip_backbone'])
    clip_model.eval()

    # load frozen CLIPN model
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    pre_train = './clipn_caches/CLIPN_ATD_Repeat2_epoch_10.pt'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clipn_model, process_train, process_test = open_clip.create_model_and_transforms(cfg['clipn_backbone'], pretrained=pre_train, device=device, freeze = False)
    clipn_model.eval()
    clipn_classifier = get_clipn_classifier(cfg, clipn_model)

    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing dataset.")

    test_tranform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((224, 224)),
    # transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])
    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])

    dataset = build_dataset(cfg['dataset'], cfg, clean_data=False) 
    val_loader = build_data_loader(cfg, data_source=dataset.val, noise_data=None, noise_class=None, img_type='original', batch_size=64, is_train=False, tfm=test_tranform, shuffle=False)
    test_loader = build_data_loader(cfg, data_source=dataset.test, noise_data=None, noise_class=None, img_type='original', batch_size=64, is_train=False, tfm=test_tranform, shuffle=False)
    train_loader = build_data_loader(cfg, data_source=dataset.train_x, noise_data=dataset._noise_data, noise_class=dataset._nciai, img_type='original', batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)        

    dalle_dataset = build_dataset(cfg['dalle_dataset'], cfg)
    dalle_train_loader = build_data_loader(cfg, data_source=dalle_dataset.train_x, noise_data=dataset._noise_data, noise_class=dataset._nciai, img_type='dalle', batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
    dalle_train_loader_F = build_data_loader(cfg, data_source=dalle_dataset.train_x, noise_data=dataset._noise_data, noise_class=dataset._nciai, img_type='dalle', batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)


    # get text features of classnames
    with open(cfg['gpt3_prompt_file']) as f:
        gpt3_prompt = json.load(f)
    print("\nGetting textual features as CLIP's classifier.")
    classname_clip_text_feas = gpt_clip_classifier(dataset.classnames, gpt3_prompt, clip_model, dataset.template)


    # train-data: image features, labels, gt_labels, zero-shot logits_yes，zero-shot logits_no
    train_data_dict = pretrain(cfg, classname_clip_text_feas, clip_model, clipn_classifier, train_loader, split='train') # data_dict = {'img_feas': img_feas, 'lbs': lbs, 'impaths': impaths, 'logits_yes': logits_yes, 'logits_no': logits_no}

    # dalle-generated-data: image features, labels, gt_labels, zero-shot logits_yes，zero-shot logits_no
    dalle_data_dict = pretrain(cfg, classname_clip_text_feas, clip_model, clipn_classifier, dalle_train_loader, split='dalle') # data_dict = {'img_feas': img_feas, 'lbs': lbs, 'impaths': impaths, 'logits_yes': logits_yes, 'logits_no': logits_no}

    # test-data: image features, labels, gt_labels, zero-shot logits_yes，zero-shot logits_no
    test_data_dict = pretrain(cfg, classname_clip_text_feas, clip_model, clipn_classifier, test_loader, split='test') # data_dict = {'img_feas': img_feas, 'lbs': lbs, 'impaths': impaths, 'logits_yes': logits_yes, 'logits_no': logits_no}


    if cfg['open_world']['is_open_world'] == True and cfg['open_world']['clean_dataset'] == True:
        cleaned_dataset = clean_model(cfg, train_data_dict, dataset, re_clean=True)
        train_loader = build_data_loader(cfg, data_source=cleaned_dataset.train_x, noise_data=dataset._noise_data, noise_class=dataset._nciai, img_type='cleaned', batch_size=256, tfm=train_tranform, is_train=True, shuffle=False) # nciai: noise_class_ids_all_incorrect

 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = Adapter(cfg).cuda()
    net = nn.DataParallel(net, device_ids=[0]) 

    run_Adapter(cfg,
            net,
            clip_model,
            clipn_classifier,
            train_loader,
            dalle_train_loader,
            test_loader,
            classname_clip_text_feas,
            train_data_dict,
            dalle_data_dict,
            test_data_dict,
            dataset
            )
                         
if __name__ == '__main__':
    main()

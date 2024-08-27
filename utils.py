from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import os
import clip
import numpy as np
import open_clip
from open_clip.transform import image_transform
import json
import random
from collections import defaultdict
import shutil

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def gpt_clip_classifier(classnames, gpt_prompts, clip_model, template):
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = []
            for t in gpt_prompts[classname]:
                texts.append(t)
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def search_hp(cfg, 
            adapter,
            img_clip_feas,
            val_labels,
            classname_clip_text_feas
            ):
    
    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        # alpha_list = [0.05,0.1,0.15,0.,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.5,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5]

        best_acc = 0
        best_alpha = 0
        with torch.no_grad():
            adapter_feas, adapter_logits = adapter(img_clip_feas) 
        init_clip_logits = 100. * img_clip_feas @ classname_clip_text_feas
        adapter_logits = logits_fuse(init_clip_logits, adapter_logits)
        
        for beta in beta_list:
            for alpha in alpha_list:
                adapter_logits = ((-1) * (beta - beta * adapter_logits)).exp()
                logits = init_clip_logits + alpha * adapter_logits  
                acc = cls_acc(logits, val_labels)      
                if acc > best_acc:
                    print("New best setting, alpha: {:.2f}; accuracy: {:.2f}".format(alpha, acc))
                    best_acc = acc
                    best_alpha = alpha  
                    best_beta = beta   

        # for alpha in alpha_list:
        #     logits = clip_logits + alpha * TeFu_logits  
        #     acc = cls_acc(logits, val_labels)      
        #     if acc > best_acc:
        #         print("New best setting, alpha: {:.2f}; accuracy: {:.2f}".format(alpha, acc))
        #         best_acc = acc
        #         best_alpha = alpha  
        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))


    return best_acc


# clip zero_shot as baseline
def logits_fuse(zero_logtis, logit, normalize='mean'):
    # This part of the code refers to paper CaFo
    # https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Prompt_Generate_Then_Cache_Cascade_of_Foundation_Models_Makes_Strong_CVPR_2023_paper.pdf
    
    # normalize logits
    logit = F.log_softmax(logit,dim=1)
    softmax_fun = nn.Softmax(dim=1)
    if normalize == 'softmax':
        zero_logtis = softmax_fun(zero_logtis)
    elif normalize =='linear':
        zero_logtis /= torch.norm(zero_logtis, p=2, dim=1, keepdim=True)
    elif normalize == 'mean':
        logits_std = torch.std(zero_logtis, dim=1, keepdim=True)
        logits_mean = torch.mean(zero_logtis, dim=1, keepdim=True)
        zero_logtis = (zero_logtis - logits_mean) / logits_std
    else:
        raise("error normalize!")
    similarity_matrix = []
    normalize_logits = []

    if normalize == 'softmax':
        current_normalize_logits = softmax_fun(logit)
    elif normalize =='linear':
        current_normalize_logits = logit / torch.norm(logit, p=2, dim=1, keepdim=True)
    elif normalize == 'mean':
        logits_std = torch.std(logit, dim=1, keepdim=True)
        logits_mean = torch.mean(logit, dim=1, keepdim=True)
        current_normalize_logits = (logit - logits_mean) / logits_std
    else:
        raise("error normalize!")
    current_similarity = current_normalize_logits * zero_logtis
    current_similarity = torch.sum(current_similarity, dim=1, keepdim=True)
    
    similarity_matrix.append(current_similarity)
    normalize_logits.append(current_normalize_logits)
    similarity_matrix = torch.stack(similarity_matrix, dim=-2)
    similarity_matrix = softmax_fun(similarity_matrix)
    normalize_logits = torch.stack(normalize_logits, dim=-2)
      
    result_logits = torch.sum(normalize_logits * similarity_matrix, dim=1)

    return result_logits


def torch_save(classifer, save_path="./"):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(classifer.cpu(), f)
        
def torch_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def pretrain(cfg, classname_clip_text_feas, clip, clipn, loader, split):
    if split == 'train':
        shots = "_" + str(cfg['shots']) + "_shots_" 
        nlb = str(cfg['open_world']['nlb']) + '_nlb_'
    elif split == 'dalle':
        shots = "_" + str(cfg['dalle_shots']) + "_dalle_shots_"
        nlb = ''
    elif split == 'test':
        shots = "_"
        nlb = ''

    clip.eval()
    clipn.fc_yes.requires_grad = False
    clipn.fc_no.requires_grad = False    
    clipn.eval()
    if cfg['load_cache'] == False:
        logits_yes_lst, probs_no_lst, img_feas_lst, lbs_lst, gt_lbs_lst, impath_lst = [], [], [], [], [], []
        pred_lbs_lst, classname_lst = [], []

        with torch.no_grad():
            for i, (imgs, lbs, gt_lbs, impath) in enumerate(tqdm(loader)):
 
                lbs = torch.where(lbs < 0, torch.tensor(int(cfg['num_classes'])-1), lbs)
                imgs, lbs, gt_lbs = imgs.cuda(), lbs.cuda(), gt_lbs.cuda()

                img_feas = clip.encode_image(imgs)
                img_feas /= img_feas.norm(dim=-1, keepdim=True)
                logits_yes = 100. * img_feas @ classname_clip_text_feas
                probs_no = clipn(imgs)

                probs_yes = F.softmax(logits_yes, dim=1)
                pred_lbs = torch.argmax(probs_yes, dim=1)

                logits_yes_lst.append(logits_yes)
                probs_no_lst.append(probs_no)
                img_feas_lst.append(img_feas)
                lbs_lst.append(lbs)
                gt_lbs_lst.append(gt_lbs)
                impath_lst.append(impath)
                # classname_lst.append(classname)

                pred_lbs_lst.append(pred_lbs)

  
        logits_yes = torch.cat(logits_yes_lst)
        probs_no = torch.cat(probs_no_lst)
        img_feas = torch.cat(img_feas_lst)
        lbs = torch.cat(lbs_lst)
        gt_lbs = torch.cat(gt_lbs_lst)
        pred_lbs = torch.cat(pred_lbs_lst)
        
        impaths = []
        for ii in range(len(impath_lst)):
            impaths.extend(impath_lst[ii])

        
        torch.save(logits_yes, cfg['cache_dir'] + "/" + split + shots + "logits_yes.pt")
        torch.save(probs_no, cfg['cache_dir'] + "/" + split + shots + "probs_no.pt")
        torch.save(img_feas, cfg['cache_dir'] + "/" + split + shots + "img_feas.pt")
        torch.save(lbs, cfg['cache_dir'] + "/" + split + shots + nlb + "lbs.pt")
        torch.save(gt_lbs, cfg['cache_dir'] + "/" + split + shots + "gt_lbs.pt")

        # file_name = cfg['cache_dir'] + "/" +  split + shots + "impaths.pt"
        # with open(file_name, 'wb') as file:
        #     pickle.dump(impaths, file)


        torch.save(pred_lbs, cfg['cache_dir'] + "/" + split + shots + "pred_lbs.pt")
        torch.save(classname_lst, cfg['cache_dir'] + "/" + split + shots + "classame.pt")
  

    else:
        logits_yes = torch.load(cfg['cache_dir'] + "/" + split + shots + "logits_yes.pt")
        probs_no = torch.load(cfg['cache_dir'] + "/" + split + shots + "probs_no.pt")
        img_feas = torch.load(cfg['cache_dir'] + "/" + split + shots + "img_feas.pt")
        lbs = torch.load(cfg['cache_dir'] + "/" + split + shots + nlb + "lbs.pt")
        gt_lbs = torch.load(cfg['cache_dir'] + "/" + split + shots + "gt_lbs.pt")

        # file_name = cfg['cache_dir'] + "/" + split + shots + "impaths.pt"
        # with open(file_name, 'rb') as file:
        #     impaths = pickle.load(file)

    data_dict = {
    'img_feas': img_feas,
    'lbs': lbs,
    'gt_lbs': gt_lbs,
    # 'impaths': impaths,
    'logits_yes': logits_yes,
    'probs_no': probs_no
}
    
    
    return data_dict
                

def get_clipn_classifier(cfg, model):

    if cfg['load_pre_clipn'] == False:
        txt = []
        model.eval()
        if cfg['num_classes']:
            path_tmp = './clipn_caches/prompt/prompt.txt'
            with open(path_tmp) as f:
                prompt_lis = f.readlines()
            num_prom = len(prompt_lis)
        

        if cfg['dataset'] == 'imagenet':
            file_name = 'imagenet_classnames.json'
        elif cfg['dataset'] == 'caltech-101':
            file_name = 'caltech101_classnames.json'
        elif cfg['dataset'] == 'oxford_pets':
            file_name = 'oxford_pets_classnames.json'
        elif cfg['dataset'] == 'dtd':
            file_name = 'dtd_classnames.json'
        elif cfg['dataset'] == 'food-101':
            file_name = 'food101_classnames.json'
        elif cfg['dataset'] == 'SUN397':
            file_name = 'SUN397_classnames.json'
        elif cfg['dataset'] == 'stanford_cars':
            file_name = 'stanford_cars_classnames.json'
        elif cfg['dataset'] == 'eurosat':
            file_name = 'eurosat_classnames.json'
        elif cfg['dataset'] == 'fgvc':
            file_name = 'fgvc_classnames.json'
        elif cfg['dataset'] == 'oxford_flowers':
            file_name = 'oxford_flowers_classnames.json'
        elif cfg['dataset'] == 'ucf101':
            file_name = 'ucf101_classnames.json'

        if cfg['dataset'] == 'fgvc':
            file_path = os.path.join(cfg['root_path'], 'fgvc_aircraft', file_name)
        else:
            file_path = os.path.join(cfg['root_path'], cfg['dataset'], file_name)

        with open(file_path, "r") as f:
            class_names = json.load(f)       
        
        for idx in range(num_prom):
            for name in class_names:
                txt.append(open_clip.tokenize(prompt_lis[idx].replace("\n", "").format(name), 77).unsqueeze(0))
        txt = torch.cat(txt, dim=0)
        txt = txt.reshape(num_prom, len(class_names), -1)
        text_inputs = txt.cuda()
        
        text_yes_ttl = torch.zeros(len(class_names), 512).cuda()
        text_no_ttl = torch.zeros(len(class_names), 512).cuda()
        
        with torch.no_grad():
            for i in range(num_prom):
                text_yes_i = model.encode_text(text_inputs[i])
                text_yes_i = F.normalize(text_yes_i, dim=-1)
                text_no_i = model.encode_text(text_inputs[i], "no")
                text_no_i = F.normalize(text_no_i, dim=-1)
                
                text_yes_ttl += text_yes_i
                text_no_ttl += text_no_i
            
        weight_no = F.normalize(text_no_ttl, dim=-1)
        weight_yes = F.normalize(text_yes_ttl, dim=-1)
        torch.save(weight_no, cfg['clipn_cache_dir'] + "/" + "weight_no.pt")
        torch.save(weight_yes, cfg['clipn_cache_dir'] + "/" + "weight_yes.pt")


        clipn_classifier =  ViT_Classifier(model.visual, weight_yes, weight_no)

        # yes_text_feas = F.normalize(clipn_classifier.fc_yes, dim=-1)
        # no_text_feas = F.normalize(clipn_classifier.fc_no, dim=-1)

        torch.save(clipn_classifier, cfg['clipn_cache_dir'] + "/" + "clipn_classifier.pt")
        # torch.save(yes_text_feas, cfg['clipn_cache_dir'] + "/text_feas/" + "yes_text_feas.pt")
        # torch.save(no_text_feas, cfg['clipn_cache_dir'] + "/text_feas/" + "no_text_feas.pt")
    
    else:
        clipn_classifier = torch.load(cfg['clipn_cache_dir'] + "/" + "clipn_classifier.pt")
        # yes_text_feas = torch.load(cfg['clipn_cache_dir'] + "/text_feas/" + "yes_text_feas.pt")
        # no_text_feas = torch.load(cfg['clipn_cache_dir'] + "/text_feas/" + "no_text_feas.pt")
            
    return clipn_classifier


class ViT_Classifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head_yes, classification_head_no):
        super().__init__()
        self.image_encoder = image_encoder
        flag = True
        self.fc_yes = nn.Parameter(classification_head_yes, requires_grad=flag)    # num_classes  num_feat_dimension
        self.fc_no = nn.Parameter(classification_head_no, requires_grad=flag)      # num_classes  num_feat_dimension
        self.scale = 100. # this is from the parameter of logit scale in CLIPN
        
    def set_frozen(self, module):
        for module_name in module.named_parameters():
            module_name[1].requires_grad = False
    def set_learnable(self, module):
        for module_name in module.named_parameters():
            module_name[1].requires_grad = True
            
    def forward(self, x):
        inputs = self.image_encoder(x)
        inputs_norm = F.normalize(inputs, dim=-1)
        fc_yes = F.normalize(self.fc_yes, dim=-1)
        fc_no = F.normalize(self.fc_no, dim=-1)
        
        logits_yes = self.scale * inputs_norm @ fc_yes.cuda().T 
        logits_no = self.scale * inputs_norm @ fc_no.cuda().T

        yesno = torch.cat([logits_yes.unsqueeze(-1), logits_no.unsqueeze(-1) ], -1)
        probs_no = torch.softmax(yesno, dim=-1)[:,:,1]
        
        return probs_no
    
    def save(self, path = "./"):
        torch_save(self, path)
        
    @classmethod
    def load(cls, filename = "./", device=None):
        return torch_load(filename, device)


def select_anchor_positive_negative(epoch, categories, cfg, dalle_data_dict, train_data_dict):

    anchor_num_per_class = cfg['anchor_num_per_class']

    dalle_lbs = dalle_data_dict['lbs']
    dalle_logits_yes = dalle_data_dict['logits_yes']
    dalle_probs_no = dalle_data_dict['probs_no']
    dalle_lbs = dalle_data_dict['lbs']

    train_pred_lbs = train_data_dict['lbs']
    train_logits_yes = train_data_dict['logits_yes']
    train_probs_no = train_data_dict['probs_no']
    train_lbs = train_data_dict['lbs']

    data_from = 'dalle_data' 
    anchor_idx_lst, positive_idx_lst, negative_idx_lst = [], [], []
    if epoch == 0: 
        
        for anchor_category in categories:
            dalle_indices = (dalle_lbs == anchor_category).nonzero().flatten()     
            dalle_logits = dalle_logits_yes[dalle_indices] 
            sorted_yes_idx = torch.argsort(dalle_logits[:,anchor_category].view(-1), descending=True)

            # 选择anchor
            max_confidence_indices = sorted_yes_idx[:anchor_num_per_class]           
            anchor_indices = dalle_indices[max_confidence_indices] 

            for anchor_idx in anchor_indices:
                anchor_idx_lst.append([data_from, torch.tensor(anchor_idx).cuda(), anchor_category])

            
            for positive_idx_tmp in sorted_yes_idx[1:(cfg['positive_num']+1)]:
                positive_idx = dalle_indices[positive_idx_tmp] 
                positive_idx_lst.append([data_from, torch.tensor(positive_idx).cuda(), anchor_category])
            
            # positive_idx_lst_ = [[data_from, dalle_indices[sorted_yes_idx[i]]] for i in range(1, 6)]
            # if positive_idx_lst != positive_idx_lst:
            #     print('00000000000')

            
            dalle_indices_negative = (dalle_lbs != anchor_category).nonzero().flatten().tolist()   
            dalle_probs_no_negative = dalle_probs_no[dalle_indices_negative] 
            dalle_lbs_negative = dalle_lbs[dalle_indices_negative]
            sorted_no_idx = torch.argsort(dalle_probs_no_negative[:,anchor_category].view(-1), descending=True)

            for negative_idx_tmp in sorted_no_idx[0:cfg['negative_num']]:
                negative_lb = dalle_lbs_negative[negative_idx_tmp]
                negative_idx = dalle_indices_negative[negative_idx_tmp] 
                negative_idx_lst.append([data_from, torch.tensor(negative_idx).cuda(), negative_lb])
            
            # negative_idx_lst_ = [[data_from, dalle_indices_negative[i]] for i in sorted_no_idx[:200]]
            # if negative_idx_lst != negative_idx_lst_:
            #     print('11111111111')

    else: 
        for anchor_category in categories:
            
            dalle_indices = (dalle_lbs == anchor_category).nonzero().flatten()      
            dalle_logits = dalle_logits_yes[dalle_indices] 
            
            train_pred_indices = (train_pred_lbs == anchor_category).nonzero().flatten() 
            train_logits = train_logits_yes[train_pred_indices] 

            logits = torch.cat([dalle_logits,train_logits],dim=0)
            probs = torch.softmax(logits, dim=0)
            sorted_yes_idx = torch.argsort(probs[:,anchor_category].view(-1), descending=True)
            
            # 选择anchor
            max_confidence_indices = sorted_yes_idx[:anchor_num_per_class]
            for max_confidence_idx in max_confidence_indices:
                if max_confidence_idx > cfg['dalle_shots'] - 1: 
                    max_confidence_idx = max_confidence_idx - cfg['dalle_shots']
                    anchor_idx = train_pred_indices[max_confidence_idx] 
                    data_from = 'train_data'
                else: 
                    anchor_idx = dalle_indices[max_confidence_idx] 
                    data_from = 'dalle_data'                
                anchor_idx_lst.append([data_from, torch.tensor(anchor_idx).cuda(), anchor_category])

            
            for positive_idx_tmp in sorted_yes_idx[1:(cfg['positive_num']+1)]:
                if positive_idx_tmp > len(dalle_indices) - 1: 
                    positive_idx_tmp = positive_idx_tmp - len(dalle_indices)
                    positive_idx = train_pred_indices[positive_idx_tmp] 
                    data_from = 'train_data'
                else: 
                    positive_idx = dalle_indices[positive_idx_tmp] 
                    data_from = 'dalle_data' 
                positive_idx_lst.append([data_from, torch.tensor(positive_idx).cuda(), anchor_category])


            # 选择20个negative样本
            dalle_indices_negative = (dalle_lbs != anchor_category).nonzero().flatten().tolist()  
            dalle_probs_no_negative = dalle_probs_no[dalle_indices_negative] 
            dalle_lbs_negative = dalle_lbs[dalle_indices_negative]

            train_pred_indices_negative = (train_pred_lbs != anchor_category).nonzero().flatten().tolist()  
            train_probs_no_negative = train_probs_no[train_pred_indices_negative] 
            train_lbs_negative = train_lbs[train_pred_indices_negative]

            no_probs = torch.cat([dalle_probs_no_negative,train_probs_no_negative],dim=0)
            sorted_no_idx = torch.argsort(no_probs[:,anchor_category].view(-1), descending=True)

            for negative_idx_tmp in sorted_no_idx[0:cfg['negative_num']]:
                if negative_idx_tmp > len(dalle_indices_negative) - 1: 
                    negative_idx_tmp = negative_idx_tmp - len(dalle_indices_negative)
                    negative_lb = train_lbs_negative[negative_idx_tmp]
                    negative_idx = train_pred_indices_negative[negative_idx_tmp] 
                    data_from = 'train_data'
                else: 
                    negative_lb = dalle_lbs_negative[negative_idx_tmp]
                    negative_idx = dalle_indices_negative[negative_idx_tmp] 
                    data_from = 'dalle_data' 
                negative_idx_lst.append([data_from, torch.tensor(negative_idx).cuda(), negative_lb])

    if anchor_num_per_class == 1:
        return anchor_idx_lst, positive_idx_lst, negative_idx_lst
    else:
        positive_idx_lst = [item for item in positive_idx_lst for _ in range(anchor_num_per_class)]
        negative_idx_lst = [item for item in negative_idx_lst for _ in range(anchor_num_per_class)]
        return anchor_idx_lst, positive_idx_lst, negative_idx_lst


def get_feas_lbs(idx_lst, categories, dalle_img_feas, img_feas, cfg, type):
    
    num_feas = dalle_img_feas.shape[1]
    num_anchors = len(categories) * cfg['anchor_num_per_class']
    num_negatives = cfg['negative_num']
    num_positives = cfg['positive_num']
   
    feas_lst, lbs_lst = [], []
    for ii in range(len(idx_lst)):
        lb = idx_lst[ii][2]
        if idx_lst[ii][0] == 'dalle_data':
            fea = dalle_img_feas[idx_lst[ii][1]]
        else:
            fea = img_feas[idx_lst[ii][1]]

        fea = fea.unsqueeze(1)

        feas_lst.append(fea.view(1,-1))
        lbs_lst.append(lb.view(1,-1)) 
    
    feas = torch.cat(feas_lst) 
    lbs = torch.cat(lbs_lst) 
    
    if type == 'negative':
        feas_new = feas.view(num_anchors, num_negatives, num_feas)
    elif type == 'positive':
        feas_new = feas.view(num_anchors, num_positives, num_feas)
    else:
        feas_new = feas.view(num_anchors, 1, num_feas)


    
    return feas_new, lbs

def get_contrastive_data(epoch, adapter, categories, cfg, dalle_data_dict, train_data_dict_tmp):
    alpha, beta = cfg['alpha'], cfg['beta']
    
    dalle_img_feas = dalle_data_dict['img_feas']
    dalle_lbs = dalle_data_dict['lbs']
    dalle_impaths = dalle_data_dict['impaths']
    dalle_logits_yes = dalle_data_dict['logits_yes']
    dalle_probs_no = dalle_data_dict['probs_no']    
    
    dalle_data_dict_tmp = {}
    adapter.eval()
    dalle_adapter_feas, dalle_adapter_logits = adapter(dalle_img_feas) 
    dalle_adapter_logits = logits_fuse(dalle_logits_yes, dalle_adapter_logits)
    dalle_adapter_logits_ = ((-1) * (beta - beta * dalle_adapter_logits)).exp()
    dalle_cls_logits = dalle_logits_yes +  alpha * dalle_adapter_logits_

    if epoch == 0: 
        
        dalle_data_dict_tmp = {
        'img_feas': dalle_adapter_feas,
        'lbs': dalle_lbs,
        'impaths': dalle_impaths,
        'logits_yes': dalle_logits_yes,
        'probs_no': dalle_probs_no
        }     
    else:
       
        
        dalle_data_dict_tmp = {
        'img_feas': dalle_adapter_feas,
        'lbs': dalle_lbs,
        'impaths': dalle_impaths,
        'logits_yes': dalle_cls_logits,
        'probs_no': dalle_probs_no
        }

    anchor_idx_lst, positive_idx_lst, negative_idx_lst = select_anchor_positive_negative(epoch, categories, cfg, dalle_data_dict_tmp, train_data_dict_tmp)

    anchor_feas, anchor_lbs = get_feas_lbs(anchor_idx_lst, categories, dalle_data_dict_tmp['img_feas'], train_data_dict_tmp['img_feas'], cfg, type='anchor')
    positive_feas, positive_lbs = get_feas_lbs(positive_idx_lst, categories, dalle_data_dict_tmp['img_feas'], train_data_dict_tmp['img_feas'], cfg, type='positive')
    negative_feas, negative_lbs = get_feas_lbs(negative_idx_lst, categories, dalle_data_dict_tmp['img_feas'], train_data_dict_tmp['img_feas'], cfg, type='negative')
    
    return anchor_feas, anchor_lbs, positive_feas, positive_lbs, negative_feas, negative_lbs

def SupConLoss(anchor_feas, anchor_lbs, positive_feas, positive_lbs, negative_feas, negative_lbs):
    
    temperature = 50
    base_temperature = 50
    contrastive_feas = torch.cat((positive_feas, negative_feas), dim=0)
    contrastive_lbs = torch.cat((positive_lbs, negative_lbs), dim=0)
    
    mask = torch.eq(anchor_lbs, contrastive_lbs.T).float()


    # compute contrastive logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feas, contrastive_feas.T),
        temperature)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits_contrastive = anchor_dot_contrast - logits_max.detach()            

    # compute log_prob
    exp_logits_contrastive = torch.exp(logits_contrastive)
    log_prob = logits_contrastive - torch.log(exp_logits_contrastive.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss_contrastive = - (temperature / base_temperature) * mean_log_prob_pos
    loss_contrastive = loss_contrastive.mean()

    return loss_contrastive

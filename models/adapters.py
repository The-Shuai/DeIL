
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import torch.nn.init as init



class Adapter(nn.Module):
      # Text-guided Fusion Adapter
      def __init__(self, cfg):
            super(Adapter, self).__init__()

      # clip_adapter = nn.Linear(clip_cache_keys.shape[0], clip_cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()


            self.adapter_layer = nn.Sequential(
                        nn.Linear(
                        in_features=1024,
                        out_features=1024,
                        bias=False),
                        nn.BatchNorm1d(1024, momentum=0.1),
                        nn.GELU())
            # self.adapter_layer = self.adapter_layer.half()
            
            self.classifier = nn.Linear(
                        in_features=1024,
                        out_features=int(cfg['num_classes']),
                        bias=False)
            # self.classifier = self.classifier.half()


            self.init_weights()

      def forward(self, fea):
            fea = fea.float()
            fea = self.adapter_layer(fea)
            logits = self.classifier(fea)
            
            return fea, logits

      def init_weights(self):
            for m in self.modules():
                  if isinstance(m, nn.Linear):
                        init.kaiming_normal_(m.weight)
                  if hasattr(m, 'bias') and m.bias is not None:
                        init.constant_(m.bias, 0)
                  elif isinstance(m, nn.BatchNorm1d):
                        init.constant_(m.weight, 1)
                        init.constant_(m.bias, 0)


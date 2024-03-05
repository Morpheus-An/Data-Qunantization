import torch
from torch import Tensor
import inspect
import warnings
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmengine.model import BaseModel, merge_dict
import torch.nn.functional as F
from mmaction.registry import MODELS
from mmaction.utils import (ConfigType, ForwardResults, OptConfigType,
                            OptSampleList, SampleList)
from mmaction.registry import MODELS
from mmaction.utils import OptSampleList

@MODELS.register_module()
class VQVAE(BaseModel, metaclass=ABCMeta):
    
    def __init__(self,
                 backbone: ConfigType,
                 codebook: ConfigType,
                 decoder: ConfigType,
                 pre_vq_conv: ConfigType,
                 post_vq_conv: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None 
                 ) -> None:
    
        if data_preprocessor is None:
            data_preprocessor = dict(type='ActionDataPreprocessor')
    
        super(VQVAE,
              self).__init__(data_preprocessor=data_preprocessor)
        
        def is_from(module, pkg_name):
            # check whether the backbone is from pkg
            model_type = module['type']
            if isinstance(model_type, str):
                return model_type.startswith(pkg_name)
            elif inspect.isclass(model_type) or inspect.isfunction(model_type):
                module_name = model_type.__module__
                return pkg_name in module_name
            else:
                raise TypeError(
                    f'Unsupported type of module {type(module["type"])}')
            
        # Record the source of the backbone.
        self.backbone_from = 'mmaction2'
        if is_from(backbone, 'mmcls.'):
            try:
                # Register all mmcls models.
                import mmcls.models  # noqa: F401
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install mmcls to use this backbone.')
            self.backbone = MODELS.build(backbone)
            self.backbone_from = 'mmcls'
        elif is_from(backbone, 'mmpretrain.'):
            try:
                # Register all mmpretrain models.
                import mmpretrain.models  # noqa: F401
            except (ImportError, ModuleNotFoundError):
                raise ImportError(
                    'Please install mmpretrain to use this backbone.')
            self.backbone = MODELS.build(backbone)
            self.backbone_from = 'mmpretrain'
        elif is_from(backbone, 'torchvision.'):
            try:
                import torchvision.models
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install torchvision to use this '
                                  'backbone.')
            self.backb_from = 'torchvision'
            self.feature_shape = backbone.pop('feature_shape', None)
            backbone_type = backbone.pop('type')
            if isinstance(backbone_type, str):
                backbone_type = backbone_type[12:]
                self.backbone = torchvision.models.__dict__[backbone_type](
                    **backbone)
            else:
                self.backbone = backbone_type(**backbone)
            # disable the classifier
            self.backbone.classifier = nn.Identity()
            self.backbone.fc = nn.Identity()
        elif is_from(backbone, 'timm.'):
            # currently, only support use `str` as backbone type
            try:
                import timm
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install timm>=0.9.0 to use this '
                                  'backbone.')
            self.backbone_from = 'timm'
            self.feature_shape = backbone.pop('feature_shape', None)
            # disable the classifier
            backbone['num_classes'] = 0
            backbone_type = backbone.pop('type')
            if isinstance(backbone_type, str):
                backbone_type = backbone_type[5:]
                self.backbone = timm.create_model(backbone_type, **backbone)
            else:
                raise TypeError(
                    f'Unsupported timm backbone type: {type(backbone_type)}')
        else:
            self.backbone = MODELS.build(backbone)
        
        # Record the source of the codebook.
        self.codebook_from = 'mmaction2'
        if is_from(codebook, 'mmcls.'):
            try:
                # Register all mmcls models.
                import mmcls.models  # noqa: F401
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install mmcls to use this codebook.')
            self.codebook = MODELS.build(codebook)
            self.codebook_from = 'mmcls'
        elif is_from(codebook, 'mmpretrain.'):
            try:
                # Register all mmpretrain models.
                import mmpretrain.models  # noqa: F401
            except (ImportError, ModuleNotFoundError):
                raise ImportError(
                    'Please install mmpretrain to use this codebook.')
            self.codebook = MODELS.build(codebook)
            self.codebook_from = 'mmpretrain'
        elif is_from(codebook, 'torchvision.'):
            try:
                import torchvision.models
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install torchvision to use this '
                                  'codebook.')
            self.codebook_from = 'torchvision'
            self.feature_shape = codebook.pop('feature_shape', None)
            codebook_type = codebook.pop('type')
            if isinstance(codebook_type, str):
                codebook_type = codebook_type[12:]
                self.codebook = torchvision.models.__dict__[codebook_type](
                    **codebook)
            else:
                self.codebook = codebook_type(**codebook)
            # disable the classifier
            self.codebook.classifier = nn.Identity()
            self.codebook.fc = nn.Identity()
        elif is_from(codebook, 'timm.'):
            # currently, only support use `str` as codebook type
            try:
                import timm
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install timm>=0.9.0 to use this '
                                  'codebook.')
            self.codebook_from = 'timm'
            self.feature_shape = codebook.pop('feature_shape', None)
            # disable the classifier
            codebook['num_classes'] = 0
            codebook_type = codebook.pop('type')
            if isinstance(codebook_type, str):
                codebook_type = codebook_type[5:]
                self.codebook = timm.create_model(codebook_type, **codebook)
            else:
                raise TypeError(
                    f'Unsupported timm codebook type: {type(codebook_type)}')
        else:
            self.codebook = MODELS.build(codebook)
        
        # Record the source of the decoder.
        self.decoder_from = 'mmaction2'
        if is_from(decoder, 'mmcls.'):
            try:
                # Register all mmcls models.
                import mmcls.models  # noqa: F401
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install mmcls to use this decoder.')
            self.decoder = MODELS.build(decoder)
            self.decoder_from = 'mmcls'
        elif is_from(decoder, 'mmpretrain.'):
            try:
                # Register all mmpretrain models.
                import mmpretrain.models  # noqa: F401
            except (ImportError, ModuleNotFoundError):
                raise ImportError(
                    'Please install mmpretrain to use this decoder.')
            self.decoder = MODELS.build(decoder)
            self.decoder_from = 'mmpretrain'
        elif is_from(decoder, 'torchvision.'):
            try:
                import torchvision.models
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install torchvision to use this '
                                  'decoder.')
            self.decoder_from = 'torchvision'
            self.feature_shape = decoder.pop('feature_shape', None)
            decoder_type = decoder.pop('type')
            if isinstance(decoder_type, str):
                decoder_type = decoder_type[12:]
                self.decoder = torchvision.models.__dict__[decoder_type](
                    **decoder)
            else:
                self.decoder = decoder_type(**decoder)
            # disable the classifier
            self.decoder.classifier = nn.Identity()
            self.decoder.fc = nn.Identity()
        elif is_from(decoder, 'timm.'):
            # currently, only support use `str` as decoder type
            try:
                import timm
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install timm>=0.9.0 to use this '
                                  'decoder.')
            self.decoder_from = 'timm'
            self.feature_shape = decoder.pop('feature_shape', None)
            # disable the classifier
            decoder['num_classes'] = 0
            decoder_type = decoder.pop('type')
            if isinstance(decoder_type, str):
                decoder_type = decoder_type[5:]
                self.decoder = timm.create_model(decoder_type, **decoder)
            else:
                raise TypeError(
                    f'Unsupported timm decoder type: {type(decoder_type)}')
        else:
            self.decoder = MODELS.build(decoder)

        # Record the source of the pre_vq_conv.
        self.pre_vq_conv_from = 'mmaction2'
        if is_from(pre_vq_conv, 'mmcls.'):
            try:
                # Register all mmcls models.
                import mmcls.models  # noqa: F401
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install mmcls to use this pre_vq_conv.')
            self.pre_vq_conv = MODELS.build(pre_vq_conv)
            self.pre_vq_conv_from = 'mmcls'
        elif is_from(pre_vq_conv, 'mmpretrain.'):
            try:
                # Register all mmpretrain models.
                import mmpretrain.models  # noqa: F401
            except (ImportError, ModuleNotFoundError):
                raise ImportError(
                    'Please install mmpretrain to use this pre_vq_conv.')
            self.pre_vq_conv = MODELS.build(pre_vq_conv)
            self.pre_vq_conv_from = 'mmpretrain'
        elif is_from(pre_vq_conv, 'torchvision.'):
            try:
                import torchvision.models
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install torchvision to use this '
                                  'pre_vq_conv.')
            self.pre_vq_conv_from = 'torchvision'
            self.feature_shape = pre_vq_conv.pop('feature_shape', None)
            pre_vq_conv_type = pre_vq_conv.pop('type')
            if isinstance(pre_vq_conv_type, str):
                pre_vq_conv_type = pre_vq_conv_type[12:]
                self.pre_vq_conv = torchvision.models.__dict__[pre_vq_conv_type](
                    **pre_vq_conv)
            else:
                self.pre_vq_conv = pre_vq_conv_type(**pre_vq_conv)
            # disable the classifier
            self.pre_vq_conv.classifier = nn.Identity()
            self.pre_vq_conv.fc = nn.Identity()
        elif is_from(pre_vq_conv, 'timm.'):
            # currently, only support use `str` as pre_vq_conv type
            try:
                import timm
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install timm>=0.9.0 to use this '
                                  'pre_vq_conv.')
            self.pre_vq_conv_from = 'timm'
            self.feature_shape = pre_vq_conv.pop('feature_shape', None)
            # disable the classifier
            pre_vq_conv['num_classes'] = 0
            pre_vq_conv_type = pre_vq_conv.pop('type')
            if isinstance(pre_vq_conv_type, str):
                pre_vq_conv_type = pre_vq_conv_type[5:]
                self.pre_vq_conv = timm.create_model(pre_vq_conv_type, **pre_vq_conv)
            else:
                raise TypeError(
                    f'Unsupported timm pre_vq_conv type: {type(pre_vq_conv_type)}')
        else:
            self.pre_vq_conv = MODELS.build(pre_vq_conv)
        
        # Record the source of the post_vq_conv.
        self.post_vq_conv_from = 'mmaction2'
        if is_from(post_vq_conv, 'mmcls.'):
            try:
                # Register all mmcls models.
                import mmcls.models  # noqa: F401
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install mmcls to use this post_vq_conv.')
            self.post_vq_conv = MODELS.build(post_vq_conv)
            self.post_vq_conv_from = 'mmcls'
        elif is_from(post_vq_conv, 'mmpretrain.'):
            try:
                # Register all mmpretrain models.
                import mmpretrain.models  # noqa: F401
            except (ImportError, ModuleNotFoundError):
                raise ImportError(
                    'Please install mmpretrain to use this post_vq_conv.')
            self.post_vq_conv = MODELS.build(post_vq_conv)
            self.post_vq_conv_from = 'mmpretrain'
        elif is_from(post_vq_conv, 'torchvision.'):
            try:
                import torchvision.models
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install torchvision to use this '
                                  'post_vq_conv.')
            self.post_vq_conv_from = 'torchvision'
            self.feature_shape = post_vq_conv.pop('feature_shape', None)
            post_vq_conv_type = post_vq_conv.pop('type')
            if isinstance(post_vq_conv_type, str):
                post_vq_conv_type = post_vq_conv_type[12:]
                self.post_vq_conv = torchvision.models.__dict__[post_vq_conv_type](
                    **post_vq_conv)
            else:
                self.post_vq_conv = post_vq_conv_type(**post_vq_conv)
            # disable the classifier
            self.post_vq_conv.classifier = nn.Identity()
            self.post_vq_conv.fc = nn.Identity()
        elif is_from(post_vq_conv, 'timm.'):
            # currently, only support use `str` as post_vq_conv type
            try:
                import timm
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install timm>=0.9.0 to use this '
                                  'post_vq_conv.')
            self.post_vq_conv_from = 'timm'
            self.feature_shape = post_vq_conv.pop('feature_shape', None)
            # disable the classifier
            post_vq_conv['num_classes'] = 0
            post_vq_conv_type = post_vq_conv.pop('type')
            if isinstance(post_vq_conv_type, str):
                post_vq_conv_type = post_vq_conv_type[5:]
                self.post_vq_conv = timm.create_model(post_vq_conv_type, **post_vq_conv)
            else:
                raise TypeError(
                    f'Unsupported timm post_vq_conv type: {type(post_vq_conv_type)}')
        else:
            self.post_vq_conv = MODELS.build(post_vq_conv)
        
        self.train_cfg = train_cfg 
        self.test_cfg = test_cfg 

    def encode(self, x, include_embeddings=False):
        h = self.pre_vq_conv(self.backbone(x))
        vq_output = self.codebook(h)
        if include_embeddings:
            return vq_output['encodings'], vq_output['embeddings']
        else:
            return vq_output['encodings']
    def predict(self, inputs:torch.Tensor):
        pass 

    def loss(self, inputs:torch.Tensor):
        """包含了原始Vqvae的三项,loss, x_recon, vq_out"""
        # print(f"in VQVAE.forward200:{inputs.shape}")
        # print(f"sample is :{inputs[0]}")
        mid = self.backbone(inputs)
        # print(f"mid.shape={mid.shape}")
        z = self.pre_vq_conv(mid)
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']))
        recon_loss = F.mse_loss(x_recon, inputs) / 0.06 

        return recon_loss, x_recon, vq_output
    
    def forward(self, inputs:torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor',
                **kwargs
        ) -> ForwardResults:
        B, num_crops, C, T, H, W = inputs.shape
        inputs = inputs.view((-1,) + inputs.shape[2:])
        if mode == 'loss':
            recon_loss, _, vq_output = self.loss(inputs)
            # print(f"predict over.")
            commitment_loss = vq_output['commitment_loss']
            loss = recon_loss + commitment_loss 
            return dict(loss=loss)  
        elif mode == 'predict':
            feats = self.backbone(inputs)
            predictions = torch.argmax(feats, 1)
            return predictions
        elif mode == 'tensor':
            return self.backbone(inputs)
            

    def decode(self, encodings):
        h = F.embedding(encodings, self.codebook.embeddings)
        h = self.post_vq_conv(self.shift_dim(h, -1, 1))
        return self.decoder(h)



    def shift_dim(self, x, src_dim=-1, dest_dim=-1, make_contiguous=True):
        n_dims = len(x.shape)
        if src_dim < 0:
            src_dim = n_dims + src_dim
        if dest_dim < 0:
            dest_dim = n_dims + dest_dim

        assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

        dims = list(range(n_dims))
        del dims[src_dim]

        permutation = []
        ctr = 0
        for i in range(n_dims):
            if i == dest_dim:
                permutation.append(src_dim)
            else:
                permutation.append(dims[ctr])
                ctr += 1
        x = x.permute(permutation)
        if make_contiguous:
            x = x.contiguous()
        return x

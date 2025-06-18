import gc
import torch
import random

from pathlib import Path
from copy import deepcopy

from fusion_algorithms.alf import ALFusionConfig, ALFusion
from utils.fusion_utils import get_num_levels
from models.base_model import BaseModel
from fusion_algorithms.hf_kf import LSFusion
from fusion_algorithms.git_rebasin import GitRebasin
from fusion_algorithms.otfusion import OTFusion
from itertools import combinations


def fuse_kf_func(models, savepath, neuron_importance_method, input_samples, labels, data=None):
    fusion = LSFusion(models)
    fused_model = fusion.fuse(
        input_samples, 
        labels, 
        fusion_method='k_means', 
        neuron_importance_method=neuron_importance_method, 
        norm_acts=False, 
        norm_neuron_importance=neuron_importance_method=='uniform'
    )
    torch.save(fused_model, savepath)
    torch.cuda.empty_cache()


def fuse_hf_func(models, savepath, neuron_importance_method, input_samples, labels, data=None):
    fusion = LSFusion(models)
    fused_model = fusion.fuse(
        input_samples, 
        labels, 
        fusion_method='hungarian', 
        neuron_importance_method=neuron_importance_method,
        norm_acts=False, 
        norm_neuron_importance=False
    )
    torch.save(fused_model, savepath)
    torch.cuda.empty_cache()


def fuse_gr_func(models, savepath, neuron_importance_method, input_samples, labels, data=None):
    git_rebasin = GitRebasin(models[0], models[1])
    fused_model = git_rebasin.fuse(input_samples, labels, neuron_importance_method=neuron_importance_method, handle_skip=True)
    torch.save(fused_model, savepath)
    torch.cuda.empty_cache()


def fuse_otf_func(models, savepath, neuron_importance_method, input_samples, labels, data=None):
    fusion = OTFusion(models[1:], models[0])
    alignment_strategy = {
        'mode': 'acts',  # 'acts' or 'wts'
        'acts': {
            'activations_dataset': input_samples,
            'activation_labels': labels,
            'neuron_importance_method': neuron_importance_method,
            'normalize_activations': False,
            'rescale_min_importance': None
        },
        'wts': {}
    }
    fused_model = fusion.fuse_models(alignment_strategy, handle_skip=True)
    torch.save(fused_model, savepath)
    torch.cuda.empty_cache()


def fuse_alf_func(models, savepath, neuron_importance_method, input_samples, labels, data=None):
    assert data is not None, "Data must be provided for ALFusion"
    fusion_config = data["fusion_config"]
    fusion_config = ALFusionConfig(**vars(fusion_config))
    fusion_config.neuron_importance_method = neuron_importance_method
    fusion = ALFusion(models)
    fused_model = fusion.fuse(fusion_config)
    torch.save(fused_model, savepath)
    del fused_model
    torch.cuda.empty_cache()

    if neuron_importance_method == 'uniform':
        config_copy = deepcopy(fusion_config)
        config_copy.skip_levels = list(range(get_num_levels(models) - 1))
        config_copy.use_kd_on_head = True
        config_copy.kd_temperature = 1.0
        config_copy.cls_head_weights = None  # they make it worse (loses ~1% accuracy in most cases)
        fusion = ALFusion(models)
        al_fused_model = fusion.fuse(config_copy)
        torch.save(al_fused_model, f'{savepath.as_posix().replace("-uniform.pt", "")}-logit-kd.pt')
        del al_fused_model
        torch.cuda.empty_cache()


def fuse_generic(models: list[BaseModel], X, y, fusion_func, additional_fusion_data, root: Path, dataset_name: str, prefix: str, num_groups: int = 0, num_per_group:int = 2, can_take_many: bool = False, skip_if_exists=True):
    num_models = len(models)
    
    if len(models) > num_per_group and num_groups > 0:
        # Fuse models pairwise

        # Generate unique model pairs
        all_pairs = list(combinations(range(num_models), num_per_group))
        num_groups = min(len(all_pairs), num_groups)  # Ensure num_groups does not exceed available groups
        sampled_groups = random.sample(all_pairs, num_groups)
        for model_idxs in sampled_groups:
            groupwise_models = [models[i] for i in model_idxs]
            model_nums = [str(i) for i in model_idxs]
            print(f"Group Fusion for {', '.join(model_nums[:-1])} and {model_nums[-1]} for {root}")

            if num_per_group == 2:
                term = 'pairwise'
            else:
                term = 'group'

            print("Uniform Scores")
            if not skip_if_exists or not (root/f'{prefix}-{term}-{'-'.join(model_nums)}-uniform.pt').exists():
                fusion_func(groupwise_models, root/f'{prefix}-{term}-{'-'.join(model_nums)}-uniform.pt', neuron_importance_method='uniform', input_samples=X, labels=y, data=additional_fusion_data)
            else:
                print(f"Skipping {str(root/f'{prefix}-{term}-{'-'.join(model_nums)}-uniform.pt')} as it already exists")

            print("Conductance-based Scores")
            if not skip_if_exists or not (root/f'{prefix}-{term}-{'-'.join(model_nums)}-conductance.pt').exists():
                fusion_func(groupwise_models, root/f'{prefix}-{term}-{'-'.join(model_nums)}-conductance.pt', neuron_importance_method='conductance', input_samples=X, labels=y, data=additional_fusion_data)   
            else:
                print(f"Skipping {str(root/f'{prefix}-{term}-{'-'.join(model_nums)}-conductance.pt')} as it already exists")

            print("DeepLIFT-based Scores")
            if not skip_if_exists or not (root/f'{prefix}-{term}-{'-'.join(model_nums)}-deeplift.pt').exists():
                fusion_func(groupwise_models, root/f'{prefix}-{term}-{'-'.join(model_nums)}-deeplift.pt', neuron_importance_method='deeplift', input_samples=X, labels=y, data=additional_fusion_data)
            else:
                print(f"Skipping {str(root/f'{prefix}-{term}-{'-'.join(model_nums)}-deeplift.pt')} as it already exists")

            gc.collect()

    # if can_take_many or num_models == 2:
    #     print("Uniform Scores")
    #     fusion_func(models, root/f'{prefix}-{num_models}-uniform.pt', neuron_importance_method='uniform', input_samples=X, labels=y, data=additional_fusion_data)

    #     print("Conductance-based Scores")
    #     fusion_func(models, root/f'{prefix}-{num_models}-conductance.pt', neuron_importance_method='conductance', input_samples=X, labels=y, data=additional_fusion_data)
        
    #     print("DeepLIFT-based Scores")
    #     fusion_func(models, root/f'{prefix}-{num_models}-deeplift.pt', neuron_importance_method='deeplift', input_samples=X, labels=y, data=additional_fusion_data)


def fuse_gr(models: list[BaseModel], root: Path, dataset_name: str, X, y, num_groups: int = 0, num_per_group:int = 2):
    fuse_generic(
        models,
        X,
        y,
        fusion_func=fuse_gr_func,
        additional_fusion_data=None,
        root=root,
        dataset_name=dataset_name,
        prefix='git-rebasin',
        num_groups=num_groups,
        num_per_group=num_per_group
    )


def fuse_hf(models: list[BaseModel], root: Path, dataset_name: str, X, y, num_groups: int = 0, num_per_group:int = 2):
    fuse_generic(
        models,
        X,
        y,
        fusion_func=fuse_hf_func,
        additional_fusion_data=None,
        root=root,
        dataset_name=dataset_name,
        prefix='hf',
        num_groups=num_groups,
        num_per_group=num_per_group
    )


def fuse_kf(models: list[BaseModel], root: Path, dataset_name: str, X, y, num_groups: int = 0, num_per_group:int = 2):
    fuse_generic(
        models,
        X,
        y,
        fusion_func=fuse_kf_func,
        additional_fusion_data=None,
        root=root,
        dataset_name=dataset_name,
        prefix='kf',
        num_groups=num_groups,
        num_per_group=num_per_group,
        can_take_many=True
    )


def fuse_otf(models: list[BaseModel], root: Path, dataset_name: str, X, y, num_groups: int = 0, num_per_group:int = 2):
    fuse_generic(
        models,
        X,
        y,
        fusion_func=fuse_otf_func,
        additional_fusion_data=None,
        root=root,
        dataset_name=dataset_name,
        prefix='otf',
        num_groups=num_groups,
        num_per_group=num_per_group,
        can_take_many=True
    )


def fuse_alf(models: list[BaseModel], 
             fusion_config: ALFusionConfig, 
             model_savedir: Path,
             dataset_name: str, num_groups: int = 0, num_per_group:int = 2):
    fuse_generic(
        models,
        fusion_config.X,
        fusion_config.y,
        fusion_func=fuse_alf_func,
        additional_fusion_data={"fusion_config": fusion_config, "kd": True},
        root=model_savedir,
        dataset_name=dataset_name,
        prefix='alf',
        num_groups=num_groups,
        num_per_group=num_per_group,
        can_take_many=True
    )
import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

from .unet import DiffusionModelUNet, DiffusionModelCondUNet
from .unetautoencoder import CondDiff

def customized_lr_scheduler(optimizer, warmup_steps=5000): # 5000 from u-vit
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'warmup':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)
    
def get_models(args):
        if 'condUNet' in args.model:
            if args.labels:
                return DiffusionModelCondUNet(
                       in_channels=args.in_channels,
                        out_channels=args.in_channels,
                        cond_channels=args.cond_channels,
                        num_channels=args.num_channels,
                        attention_levels=args.attention_levels,
                        num_res_blocks=args.num_res_blocks,
                        num_head_channels=args.num_head_channels, 
                        with_conditioning=True,
                        learn_sigma=args.learn_sigma,
                    )
            return DiffusionModelCondUNet(
                       in_channels=args.in_channels,
                        out_channels=args.in_channels,
                        cond_channels=args.cond_channels,
                        num_channels=args.num_channels,
                        attention_levels=args.attention_levels,
                        num_res_blocks=args.num_res_blocks,
                        num_head_channels=args.num_head_channels, 
                        with_conditioning=False,
                        learn_sigma=args.learn_sigma,
            )
        elif 'UNet' in args.model:
            if args.labels:
                return DiffusionModelUNet(
                    in_channels=args.in_channels,
                    out_channels=args.in_channels,
                    num_channels=args.num_channels,
                    attention_levels=args.attention_levels,
                    num_res_blocks=args.num_res_blocks,
                    num_head_channels=args.num_head_channels, 
                    with_conditioning=True,
                    learn_sigma=args.learn_sigma,
                    )
            return DiffusionModelUNet(
                    in_channels=args.in_channels,
                    out_channels=args.in_channels,
                    num_channels=args.num_channels,
                    attention_levels=args.attention_levels,
                    num_res_blocks=args.num_res_blocks,
                    num_head_channels=args.num_head_channels, 
                    with_conditioning=False,
                    learn_sigma=args.learn_sigma,
                    )
        
        elif 'conddiff' in args.model:
            return CondDiff(
                in_channels=args.in_channels,
                out_channels=args.in_channels,
                num_channels=args.num_channels,
                attention_levels=args.attention_levels,
                num_res_blocks=args.num_res_blocks,
                num_head_channels=args.num_head_channels, 
                with_conditioning=False,
                learn_sigma=args.learn_sigma,
                )
        else:
            raise '{} Model Not Supported!'.format(args.model)
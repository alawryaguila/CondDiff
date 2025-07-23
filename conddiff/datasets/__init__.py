from .una_datasets import UNAGen, UNAModule
from .precalc_datasets import Precalc, precalcModule
from .baugh_datasets import BaughBL, BaughBLModule
def get_dataset(args, device='cpu'):

    if args.dataset == 'una':
        return UNAGen(args.data_config_path, training_=args.training, device=device)
    elif args.dataset == 'precalc':
        return Precalc(args.data_config_path, training_=args.training)
    elif args.dataset == 'baughBL':
        return BaughBL(args.data_config_path, training_=args.training)
    else:
        raise NotImplementedError(args.dataset)
    

def get_datamodule(args):
    if args.dataset == 'precalc':
        return precalcModule(args)
    elif args.dataset == 'una':
        return UNAModule(args)   
    elif args.dataset == 'baughBL':
        return BaughBLModule(args)
    else:
        raise NotImplementedError(args.dataset)

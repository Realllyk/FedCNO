from models.ClassiFilerNet import ClassiFilerNet
from models.CGE_Variants import CGEVariant
from models.MANDO_Net import MANDONet


def build_model(args, input_size, time_steps):
    if args.model_type == "CBGRU":
        return ClassiFilerNet(input_size, time_steps)
    if args.model_type == "CGE":
        return CGEVariant()
    if args.model_type == "MANDO":
        return MANDONet()
    raise ValueError(f"Unsupported model_type: {args.model_type}")


def get_local_lr(args):
    if args.model_type == "CBGRU":
        return args.cbgru_local_lr
    if args.model_type == "CGE":
        return args.cge_local_lr
    if args.model_type == "MANDO":
        return args.mando_local_lr
    raise ValueError(f"Unsupported model_type: {args.model_type}")


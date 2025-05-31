from args import parse_train_opt
from DGSDP import DGSDP


def train(opt):
    model = DGSDP(opt.feature_type)
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)

from trainer import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()

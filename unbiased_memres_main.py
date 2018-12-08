from datagens.cocobbox import CocoBboxDatagen
from models.segnet import SegNet
from trainers.cocounbiasedbbox_trainer import Trainer
import resource
import sys
import torchvision.models as models
import yaml


def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 4, hard))


def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


def main(config, checkpoint_path):
    datagen = CocoBboxDatagen(config)
    vgg16 = models.vgg16(pretrained=True)
    model = SegNet(2, 3)
    model.init_vgg16_params(vgg16)
    trainer = Trainer(datagen, model, config)
    trainer.run(checkpoint_path)


if __name__ == "__main__":
    memory_limit()
    try:
        if len(sys.argv) < 2:
            print("Usage: python main.py <path to config file>"
                  "<OPTIONAL: path to checkpoint>")
            exit()

        config_path = sys.argv[1]
        with open(config_path, 'r') as ymlfile:
            config = yaml.load(ymlfile)

        checkpoint_path = None
        if len(sys.argv) > 2:
            checkpoint_path = sys.argv[2]

        main(config, checkpoint_path)
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)

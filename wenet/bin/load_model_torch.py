from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.train_utils import (
    add_model_args)
import fire

def get_args():
    parser = argparse.ArgumentParser(description='load model in torch')

    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--checkpoint', help='checkpoint model')

    args = parser.parse_args()
    print(args)
    return args

if __name__ == '__main__':
    fire.Fire()
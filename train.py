import argparse
import random
import numpy as np
import torch
from utils import trainer


# def main(config):
    # val_f1_score = {
    #     category: validate('val', device, loader, encoder, decoder,
    #                        train_loader.dataset.vocab, f'{category}.json', config['categories'])
    #     for category, loader in zip(config['categories'], val_loader)
    # }
    # test_f1_score = {
    #     category: validate('test', device, loader, encoder, decoder,
    #                        train_loader.dataset.vocab, f'{category}.json', config['categories'])
    #     for category, loader in zip(config['categories'], test_loader)
    # }
    # print(f'Validation F1 score: {val_f1_score}')
    # print(f'Test F1 score: {test_f1_score}')
    # val_results = {
    #     category: evaluate('val', category, f'{category}.json')
    #     for category in config['categories']
    # }
    # test_results = {
    #     category: evaluate('test', category, f'{category}.json')
    #     for category in config['categories']
    # }
    # print(f'Validation results: {val_results}')
    # print(f'Test results: {test_results}')

def parse_args():
    parser = argparse.ArgumentParser(description="Train")

    parser.add_argument('--tag',default='default', type=str, help="Tag for the experiment")
    parser.add_argument('--batch-size', default=128,type=int, help='Training batch size')
    parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--epochs', default=5, type=int, help="Max training epochs")
    parser.add_argument('--embed-size', default=256, type=int, help="Size of image and word embeddings")
    parser.add_argument('--hidden-size', default=512, type=int, help="Dimensionality of features in hidden state of the decoder")
    parser.add_argument('--encoder', default='resnet50', type=str, help="Type of ResNet Encoder")
    parser.add_argument('--vocab-threshold', default=5, type=int, help="Minimum word count threshold")
    parser.add_argument('--num-workers', default=0, type=int, help="Number of workers for data loading")
    parser.add_argument('--num-layers', default=1, type=int, help="Number of layers in LSTM")
    parser.add_argument('--save-every',  default=1, type=int, help="Frequency of saving model weights")
    parser.add_argument('--print-every', default=100, type=int, help="Frequency of printing training statistics")
    parser.add_argument('--max_len',default=25, type=int, help="Maximum length of caption (in words)")
    parser.add_argument('--resume',default=False, type=bool, help="Resume training from checkpoint")
    parser.add_argument('--encoder_path',default=None, type=str, help="Path to checkpoint to resume encoder")
    parser.add_argument('--decoder_path',default=None, type=str, help="Path to checkpoint to resume decoder")
    args = parser.parse_args()
    return args

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    setup_seed(2023)
    args = parse_args()
    trainer = trainer.trainer(args)
    trainer.setup()
    #trainer.train()
    #trainer.val()
    trainer.cats_val()

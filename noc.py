#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/5/30 23:19
# @File     : noc.py
# @Project  : lab
import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils import data
from tqdm import trange

import utils
import model


def train(device, data_loader, encoder, decoder, criterion, optimizer, vocab_size, total_step, log_file):
    # Open the training log file.
    f = open(os.path.join('log/', log_file), 'w')

    for epoch in range(1, args.epochs + 1):

        for step in trange(1, total_step + 1):
            # Randomly sample a caption length, and sample indices with that length.
            indices = data_loader.dataset.get_train_indices()
            # Create and assign a batch sampler to retrieve a batch with the sampled indices.
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            data_loader.batch_sampler.sampler = new_sampler

            # Obtain the batch.
            images, captions = next(iter(data_loader))

            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(device)
            captions = captions.to(device)

            # Zero the gradients.
            decoder.zero_grad()
            encoder.zero_grad()

            # Pass the inputs through the CNN-RNN model.
            features = encoder(images)
            outputs = decoder(features, captions)

            # Calculate the batch loss.
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

            # Backward pass.
            loss.backward()

            # Update the parameters in the optimizer.
            optimizer.step()

            # Get training statistics.
            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (
                epoch, args.epochs, step, total_step, loss.item(), np.exp(loss.item())
            )

            # Print training statistics (on same line).
            print('\r' + stats, end="")
            sys.stdout.flush()

            # Print training statistics to file.
            f.write(stats + '\n')
            f.flush()

            # Print training statistics (on different line).
            if step % args.print_every == 0:
                print('\r' + stats)

        # Save the weights.
        if epoch % args.save_every == 0:
            torch.save(encoder.state_dict(), os.path.join('checkpoints/', 'encoder-%d.pkl' % epoch))
            torch.save(decoder.state_dict(), os.path.join('checkpoints/', 'decoder-%d.pkl' % epoch))

    # Close the training log file.
    f.close()


def main(config):
    device = utils.reproduce(args.seed)
    train_loader = utils.get_loader(
        mode='train', batch_size=args.batch_size,
        vocab_threshold=args.vocab_threshold, vocab_from_file=args.vocab_from_file
    )
    val_loader = [
        utils.get_loader(mode='val', batch_size=args.batch_size, category=category)
        for category in config['categories']
    ]
    test_loader = [
        utils.get_loader(mode='test', batch_size=args.batch_size, category=category)
        for category in config['categories']
    ]
    encoder = model.EncoderResNet(args.embed_size, args.resnet).to(device)
    decoder = model.DecoderRNN(
        args.embed_size, args.hidden_size, vocab_size := len(train_loader.dataset.vocab), args.num_layers
    ).to(device)
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(encoder.embed.parameters()), lr=args.lr)
    total_step = int(len(train_loader.dataset.caption_lengths) / train_loader.batch_sampler.batch_size) + 1
    train(device, train_loader, encoder, decoder, criterion, optimizer, vocab_size, total_step, 'train.log')


if __name__ == '__main__':
    with open('config/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    parser = argparse.ArgumentParser(description="Image Captioning of Novel Objects by PyTorch")
    parser.add_argument('--seed', '-s', default=config['seed'], type=int, help="Set random seed")
    parser.add_argument('--batch-size', '-b', type=int, default=config['batch_size'], help='Training batch size')
    parser.add_argument('--lr', default=config['lr'], type=float, help="Learning rate")
    parser.add_argument('--epochs', '-e', default=config['epochs'], type=int, help="Max training epochs")
    parser.add_argument('--embed-size', default=config['embed'], type=int, help="Size of image and word embeddings")
    parser.add_argument('--hidden-size', default=config['hidden'], type=int, help="Dimensionality of features in "
                                                                                  "hidden state of the decoder")
    parser.add_argument('--resnet', '-r', default=config['resnet'], type=str, help="Type of ResNet Encoder")
    parser.add_argument('--vocab-threshold', '-t', default=config['vt'], type=int, help="Minimum word count threshold")
    parser.add_argument('--vocab-from-file', '-v', action='store_true', help="Load vocab from existing vocab file")
    parser.add_argument('--num-layers', '-n', default=config['num_layers'], type=int, help="Number of layers in LSTM")
    parser.add_argument('--save-every', '-s', default=config['save_every'], type=int, help="Frequency of saving model "
                                                                                           "weights")
    parser.add_argument('--print-every', '-p', default=config['print_every'], type=int, help="Frequency of printing "
                                                                                             "training statistics")
    args = parser.parse_args()
    main(config)

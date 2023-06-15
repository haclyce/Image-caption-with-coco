import numpy as np
import torch
from matplotlib import pyplot as plt


def clean_sentence(data_loader, output):
    sentence = ""
    for i in output:
        word = data_loader.dataset.vocab.idx2word[i]
        if word == data_loader.dataset.vocab.start_word:
            continue
        if word == data_loader.dataset.vocab.end_word:
            break
        else:
            sentence = sentence + " " + word
    return sentence


def get_prediction(data_loader, device, encoder, decoder):
    orig_image, image = next(iter(data_loader))
    plt.imshow(np.squeeze(orig_image))
    plt.title('Sample Image')
    plt.show()
    image = image.to(device)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)
    sentence = clean_sentence(data_loader, output)
    print(sentence)


def generate_caption(image, encoder, decoder, vocab):
    with torch.no_grad():
        feature = encoder(image).unsqueeze(1)
        sampled_ids = decoder.sample(feature)
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == '<end>':
            break
        sampled_caption.append(word)
    sentence = ' '.join(sampled_caption[1:])
    return sentence

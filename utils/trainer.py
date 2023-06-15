import json
import os
import torch
import model
from torch import nn
from tqdm import tqdm
from utils.data_loader import get_loader
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision import transforms
#import time

class trainer(object):
    def __init__(self, args):
        self.args = args

        if not os.path.isdir('./log'):
            os.mkdir('./log')
        if not os.path.isdir(f'./checkpoints/{args.tag}'):
            os.mkdir(f'./checkpoints/{args.tag}')
        if not os.path.isdir('./vocab'):
            os.mkdir('./vocab')
        if not os.path.isdir('./results'):
            os.mkdir('./results')
    def setup(self):

        args=self.args
        categories_list = ['bottle','bus','couch','microwave','pizza','racket','suitcase','zebra']
        self.categories = {i:categories_list[i] for i in range(len(categories_list))}

        # examine if cuda is available
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            raise Exception('No GPU found, please run without --cuda')
        
        print(f"Using device: {self.device}")
        transform_train = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))])
        
        transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))])
        # Create the data loader.
        self.transform_val = transform_val
        self.train_loader = get_loader(mode='train', batch_size=args.batch_size,num_workers=args.num_workers,
                                       transform=transform_train,max_len=args.max_len)
        
        self.val_loader = get_loader(mode='val', batch_size=1,num_workers=args.num_workers,
                                     transform=transform_val,max_len=args.max_len)
        
        self.test_loader = get_loader(mode='test', batch_size=1,num_workers=args.num_workers,
                                     transform=transform_val,max_len=args.max_len)
        self.vocab_size = len(self.train_loader.dataset.vocab)
        # Build the models
        self.encoder = model.encoder.EncoderResNet(args.embed_size, args.encoder).to(self.device)
        self.decoder = model.decoder.DecoderRNN(
            args.embed_size, args.hidden_size, self.vocab_size, args.num_layers).to(self.device)
        
        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(params=list(self.decoder.parameters()) + list(self.encoder.parameters()),lr=args.lr)
        self.total_step = len(self.train_loader)
        

    def train(self):
        # Open the training log file.
        args= self.args
        f = open(os.path.join('log/', f"{args.tag}_train.log"), 'w')

        for epoch in range(1, args.epochs + 1):
            for step, (images, captions,id) in enumerate(tqdm(self.train_loader)):
                images = images.to(self.device)
                captions = captions.to(self.device)

                # Zero the gradients.
                self.decoder.zero_grad()
                self.encoder.zero_grad()
                
                # Pass the inputs through the CNN-RNN model.
                features = self.encoder(images)
                outputs = self.decoder(features, captions)
                
                # Calculate the batch loss.
                
                loss = self.loss_fn(outputs.view(-1, self.vocab_size), captions.view(-1))
                                
                # Backward pass.
                self.optimizer.zero_grad()
                loss.backward()

                # Update the parameters in the optimizer.
                self.optimizer.step()                
                # Get training statistics.
                
                if (step+1) % args.print_every == 0:
                    stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (epoch, args.epochs, step+1, self.total_step, loss.item())
                    f.write(stats + '\n')
            # Save the weights.
            if epoch % args.save_every == 0:
                torch.save(self.encoder.state_dict(), os.path.join(f'./checkpoints/{args.tag}', f'encoder-{epoch}.pth'))
                torch.save(self.decoder.state_dict(), os.path.join(f'./checkpoints/{args.tag}', f'decoder-{epoch}.pth'))

        # Close the training log file.
        f.close()


    def val(self):
        args = self.args
        if args.resume:
            self.encoder.load_state_dict(torch.load(args.encoder_path))
            self.decoder.load_state_dict(torch.load(args.decoder_path))
        self.encoder.eval(), self.decoder.eval()
        #if the result dir does not exist, create one
        if not os.path.isdir(f'./results/{args.tag}'):
            os.mkdir(f'./results/{args.tag}')
        #stores the results
        result_dir=os.path.join(f'./results/{args.tag}','caption_results.json')
        caption_results = []
        #stores the scores
        scores_dir=os.path.join(f'./results/{args.tag}','scores.json')
        scores = []
        #the category dir for each image
        category_dir='./data/coco/annotations/instances_val2014.json'
        coco_cat=COCO(category_dir)
        #compute F1 scores
        tp=0
        fp_fn=0
        image_ids = []
        for (images, captions,img_ids) in tqdm(self.val_loader):
            images = images.to(self.device)
            if img_ids[0].item() in image_ids:
                continue
            image_ids.append(img_ids[0].item())

            # calculates the caption from the image
            pred_caption =generate_caption(images,self.encoder,self.decoder, self.train_loader.dataset.vocab)
            img_ids=img_ids[0].item()
            #find the pictrue's category
            
            anns=coco_cat.loadAnns(coco_cat.getAnnIds(imgIds=img_ids))
            category_ids=set()
            for i in range(len(anns)):
                category_ids.add(coco_cat.cats[anns[i]['category_id']]['name'])
            #print(category_ids)
            #calculate the F1 score
            flag=False
            for category_id in category_ids:
                if category_id in pred_caption:
                    flag=True
                    break
            if flag:
                tp+=1
            else:
                fp_fn+=1
            #ground_truth=generate_ground_truths(captions,self.train_loader.dataset.vocab)
            caption_results.append({'image_id':int(img_ids),'caption':pred_caption})
        
        # write results to file
        with open((result_dir), 'w') as f:
            json.dump(caption_results, f)
        #calculates the scores
        coco=self.val_loader.dataset.coco
        cocoRes=coco.loadRes(result_dir)
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.evaluate()
        f1_scores=2*tp/(2*tp+fp_fn)
        for metric, score in cocoEval.eval.items():
            scores.append({'metric':metric,'score':score})
        scores.append({'metric':'F1','score':f1_scores})

        with open((scores_dir), 'w') as f:
            json.dump(scores, f)

    def cats_val(self):
        args = self.args
        #the category dir for each image
        category_dir='./data/coco/annotations/instances_val2014.json'
        coco_cat=COCO(category_dir)
        if args.resume:
            self.encoder.load_state_dict(torch.load(args.encoder_path))
            self.decoder.load_state_dict(torch.load(args.decoder_path))
        self.encoder.eval(), self.decoder.eval()
        for i in range(len(self.categories)): 
            category=self.categories[i]
            mode=category+'_val'
            if not os.path.isdir(f'./results/{args.tag}'):
                os.mkdir(f'./results/{args.tag}')

            #stores the results
            result_dir=os.path.join(f'./results/{args.tag}',f'{category}_caption_results.json')
            caption_results = []
            #stores the scores
            scores_dir=os.path.join(f'./results/{args.tag}',f'{category}_scores.json')
            scores = []

            #compute F1 scores
            tp=0
            fp_fn=0
            image_ids = []
            dataloader=get_loader(mode=mode,batch_size=1,num_workers=args.num_workers,transform=self.transform_val)
            for (images, captions,img_ids) in tqdm(dataloader):
                images = images.to(self.device)
                if img_ids[0].item() in image_ids:
                    continue
                image_ids.append(img_ids[0].item())

                # calculates the caption from the image
                pred_caption =generate_caption(images,self.encoder,self.decoder, self.train_loader.dataset.vocab)
                img_ids=img_ids[0].item()
                #find the pictrue's category
                
                anns=coco_cat.loadAnns(coco_cat.getAnnIds(imgIds=img_ids))
                category_ids=set()
                for i in range(len(anns)):
                    category_ids.add(coco_cat.cats[anns[i]['category_id']]['name'])
                #print(category_ids)
                #calculate the F1 score
                flag=False
                for category_id in category_ids:
                    if category_id in pred_caption:
                        flag=True
                        break
                if flag:
                    tp+=1
                else:
                    fp_fn+=1
                #ground_truth=generate_ground_truths(captions,self.train_loader.dataset.vocab)
                caption_results.append({'image_id':int(img_ids),'caption':pred_caption})
            
            # write results to file
            with open((result_dir), 'w') as f:
                json.dump(caption_results, f)
            #calculates the scores
            coco=dataloader.dataset.coco
            cocoRes=coco.loadRes(result_dir)
            cocoEval = COCOEvalCap(coco, cocoRes)
            cocoEval.evaluate()
            f1_scores=2*tp/(2*tp+fp_fn)
            for metric, score in cocoEval.eval.items():
                scores.append({'metric':metric,'score':score})
            scores.append({'metric':'F1','score':f1_scores})
            
            with open((scores_dir), 'w') as f:
                json.dump(scores, f)


# def generate_ground_truths(caption, vocab):
#     caption=caption.squeeze(0).tolist()
#     for i in range(len(caption)):
#         caption[i]=vocab.idx2word[caption[i]]
#     sentence=' '.join(caption[1:])
#     if '<end>' in sentence:
#         sentence=sentence.split('<end>')[0]
#     return sentence

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

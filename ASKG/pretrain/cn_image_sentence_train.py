import os
import opts
import pickle
import cv2

import eval_utils
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from collections import OrderedDict
from models.cnn import Densenet121_AG, Fusion_Branch
from models.SentenceImageModel import *
from misc.utils import NoamOpt
from read_cn_data2 import get_loader_cn
from read_fh21_data import get_loader2
from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score
import torchvision.transforms as transforms

import codecs
import sys
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
   transforms.Resize((256, 256)),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize,
])


def Attention_gen_patchs(ori_image, fm_cuda):
    # fm => mask =>(+ ori-img) => crop = patchs
    feature_conv = fm_cuda.data.cpu().numpy()
    size_upsample = (224, 224) 
    bz, nc, h, w = feature_conv.shape

    patchs_cuda = torch.FloatTensor().cuda()

    for i in range(0, bz):
        feature = feature_conv[i]
        cam = feature.reshape((nc, h*w))
        cam = cam.sum(axis=0)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)

        heatmap_bin = binImage(cv2.resize(cam_img, size_upsample))
        heatmap_maxconn = selectMaxConnect(heatmap_bin)
        heatmap_mask = heatmap_bin * heatmap_maxconn

        ind = np.argwhere(heatmap_mask != 0)
        minh = min(ind[:, 0])
        minw = min(ind[:, 1])
        maxh = max(ind[:, 0])
        maxw = max(ind[:, 1])
        
        # to ori image 
        image = ori_image[i].numpy().reshape(224, 224, 3)
        image = image[int(224*0.334):int(224*0.667), int(224*0.334):int(224*0.667), :]

        image = cv2.resize(image, size_upsample)
        image_crop = image[minh:maxh, minw:maxw, :] * 256 # because image was normalized before
        image_crop = preprocess(Image.fromarray(image_crop.astype('uint8')).convert('RGB')) 

        img_variable = torch.autograd.Variable(image_crop.reshape(3, 224, 224).unsqueeze(0).cuda())

        patchs_cuda = torch.cat((patchs_cuda, img_variable), 0)

    return patchs_cuda


def binImage(heatmap):
    _, heatmap_bin = cv2.threshold(heatmap, 178, 255, cv2.THRESH_BINARY)
    return heatmap_bin


def selectMaxConnect(heatmap):
    labeled_img, num = label(heatmap, connectivity=2, background=0, return_num=True)    
    max_label = 0
    max_num = 0
    for i in range(1, num+1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)
    if max_num == 0:
       lcc = (labeled_img == -1)
    lcc = lcc + 0
    return lcc 

def abstracts_train():
    model.train()   
    tqdm_bar = tqdm(train_loader, desc="Training")

    for iteration, (ids, abstracts, abstracts_labels, medterm_labels) in enumerate(tqdm_bar):
        abstracts = abstracts.to(device)
        abstracts_labels = abstracts_labels.to(device)
        medterm_labels = medterm_labels.to(device)

        rnn_NoamOpt.zero_grad()

        med_probs, abstracts_outputs = model(att_feats=abstracts, input_ids=abstracts, input_type='sentence')
        med_loss = med_crit(med_probs, medterm_labels)

        caption_loss = outputs_crit(abstracts_outputs.view(-1, abstracts_outputs.size(-1)), abstracts_labels.view(-1))
        loss = 2.0 * med_loss + caption_loss
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        rnn_NoamOpt.step()

        train_loss = loss.item()

        tqdm_bar.desc = "train_loss = {:.5f}, med_loss = {:.5f}, caption_loss = {:.5f},"\
                        "lr = {:.5f}, cnn_lr = {:.5f}"\
            .format(train_loss, med_loss, caption_loss,
                    rnn_NoamOpt.optimizer.param_groups[0]['lr'], opt.cnn_learning_rate)

def images_train():
    cnn_model.train()
    aux_model.train()
    fusion_model.train()
    model.train()
    total_step = len(img_train_loader)

    tqdm_bar = tqdm(img_train_loader, desc="Training")

    for iteration, (ids, image_id, imgs, input_ids, lm_labels, medterm_labels) in enumerate(tqdm_bar):

        imgs = imgs.to(device)
        input_ids = input_ids.to(device)
        lm_labels = lm_labels.to(device)
        medterm_labels = medterm_labels.to(device)

        rnn_NoamOpt.zero_grad()
        cnn_optimizer.zero_grad()

        # compute output
        output_global, fm_global, pool_global = cnn_model(imgs)
        
        patchs_var = Attention_gen_patchs(input, fm_global)

        output_local, _, pool_local = aux_model(patchs_var)
        #print(fusion_var.shape)
        output_fusion = fusion_model(pool_global, pool_local)

        med_porbs, findings_outputs = model(att_feats = output_fusion, input_ids = input_ids, input_type = 'img')

        med_loss = med_crit(med_porbs, medterm_labels)
        caption_loss = outputs_crit(findings_outputs.view(-1, findings_outputs.size(-1)), lm_labels.view(-1))
        loss = 2.0*med_loss + caption_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        nn.utils.clip_grad_norm_(cnn_model.parameters(), opt.grad_clip)

        rnn_NoamOpt.step()
        cnn_optimizer.step()

        train_loss = loss.item()

        # print("iter {}/{} (epoch {}/{}), train_loss = {:.5f}, med_loss = {:.5f}, caption_loss = {:.5f},"
        #       "lr = {:.5f}, cnn_lr = {:.5f}, time/batch={:.3f}"
        #       .format(iteration, total_step, epoch, opt.max_epochs,
        #               train_loss, med_loss, caption_loss,
        #               rnn_NoamOpt.optimizer.param_groups[0]['lr'], opt.cnn_learning_rate, end - start))

        tqdm_bar.desc = "train_loss = {:.5f}, med_loss = {:.5f}, caption_loss = {:.5f},"\
                        "lr = {:.5f}, cnn_lr = {:.5f}"\
            .format(train_loss, med_loss, caption_loss,
                    rnn_NoamOpt.optimizer.param_groups[0]['lr'], opt.cnn_learning_rate)

def modify_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' not in k:
            k = 'module.' + k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k] = v
    return new_state_dict

def load_best_model():
    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
    cnn_checkpoint_path = os.path.join(opt.checkpoint_path, 'model-cnn-best.pth')
    caption_state_dict = torch.load(checkpoint_path)
    cnn_state_dict = torch.load(cnn_checkpoint_path)
    new_state_dict = modify_state_dict(caption_state_dict)
    model.load_state_dict(new_state_dict)

    new_state_dict = modify_state_dict(cnn_state_dict)
    cnn_model.load_state_dict(new_state_dict)

def load_model():
    checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
    cnn_checkpoint_path = os.path.join(opt.checkpoint_path, 'model-cnn.pth')
    caption_state_dict = torch.load(checkpoint_path)
    cnn_state_dict = torch.load(cnn_checkpoint_path)
    new_state_dict = modify_state_dict(caption_state_dict)
    model.load_state_dict(new_state_dict)

    new_state_dict = modify_state_dict(cnn_state_dict)
    cnn_model.load_state_dict(new_state_dict)

def update_encoder_model(path):
    encoder_cnn_path = os.path.join(opt.encoder_path, 'model-cnn-best.pth')
    encoder_rnn_path = os.path.join(opt.encoder_path, 'model-best.pth')
    cnn_state_dict = torch.load(encoder_cnn_path)
    cnn_state_dict = modify_state_dict(cnn_state_dict)
    cnn_model.load_state_dict(cnn_state_dict)

    rnn_state_dict = torch.load(encoder_rnn_path)
    rnn_state_dict = modify_state_dict(rnn_state_dict)

    model_state_dict = model.state_dict()
    model_state_dict = modify_state_dict(model_state_dict)

    rnn_state_dict = {k:v for k,v in rnn_state_dict.items() if k in model_state_dict.keys()}
    print("updated", rnn_state_dict.keys())

    model_state_dict.update(rnn_state_dict)
    model.load_state_dict(model_state_dict)


def train():
    #best_val_score = None
    best_val_score = eval()['CIDEr']
    
    for epoch in trange(int(opt.max_epochs), desc="Epoch"):
    #for epoch in trange(int(5), desc="Epoch"):
        #abstracts_train()
        images_train()

        if epoch % opt.val_every_epoch == 0:
            lang_stats = eval()

            current_score = lang_stats['CIDEr']

            best_flag = False
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True
            checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print("model saved to {}".format(checkpoint_path))
            
            cnn_checkpoint_path = os.path.join(opt.checkpoint_path, 'model-cnn.pth')
            torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
            print("cnn model saved to {}".format(cnn_checkpoint_path))

            if best_flag:
                print(best_val_score)
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                # optim_checkpoint_path = os.path.join(opt.checkpoint_path, 'rnn_optimizer-best.pth')
                # torch.save(rnn_NoamOpt.optimizer.state_dict(), optim_checkpoint_path)
                # torch.save(rnn_NoamOpt.state_dict(), optim_checkpoint_path)
                # print("model saved to {}".format(optim_checkpoint_path))

                cnn_checkpoint_path = os.path.join(opt.checkpoint_path, 'model-cnn-best.pth')
                torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
                print("cnn model saved to {}".format(cnn_checkpoint_path))
                # cnn_optim_checkpoint_path = os.path.join(opt.checkpoint_path, 'cnn_optimizer-best.pth')
                # torch.save(cnn_optimizer.state_dict(), cnn_optim_checkpoint_path)
                # print("model saved to {}".format(cnn_optim_checkpoint_path))

def decode_transformer_findings(idw2word, sampled_findings):
    decode_list = []
    n_samples, n_words = sampled_findings.size()

    for n in range(n_samples):
        decoded = []
        words = []
        for i in range(n_words):
            token = idw2word[int(sampled_findings[n][i])]
            if token == '<BOS>':
                continue
            if token == '<EOS>':
                break
            if token != '<UNK>' and token != '<BLANK>':
                words.append(token)
        if len(words) != 0:
            decoded.append(' '.join(words))
        decode_list.append(' '.join(decoded))
    return decode_list  # [batch_size, length]

def eval():
    ## TODO
    
    tqdm_bar = tqdm(img_valid_loader, desc="Evaluating")

    cnn_model.eval()
    aux_model.eval()
    fusion_model.eval()

    model.eval()

    num_show = 0

    id2findings_t = {}  # true
    id2findings_g = {}  # generated

    id2captions_t = {}
    id2captions_g = {}

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()

    count = 0
    with torch.no_grad():
        for iteration, (ids, image_ids, imgs, input_ids, lm_labels, medterm_labels) in enumerate(tqdm_bar):
            # imgs = [[img.to(device) for img in sample] for sample in imgs]
            imgs = imgs.to(device)
            medterm_labels = medterm_labels.to(device)

            output_global, fm_global, pool_global = cnn_model(imgs)
        
            patchs_var = Attention_gen_patchs(input, fm_global)

            output_local, _, pool_local = aux_model(patchs_var)
            #print(fusion_var.shape)
            output_fusion = fusion_model(pool_global, pool_local)

            med_porbs, findings_seq = model(att_feats=output_fusion, mode='inference', input_type='img')
            #if len(findings_seq) > 512:
            #    findings_seq = findings_seq[:511]
            findings_samples = decode_transformer_findings(img_train_dataset.idw2word, findings_seq)
            findings_truths = decode_transformer_findings(img_train_dataset.idw2word, lm_labels)
            # print("label", lm_labels)
            # print(lm_labels)

            for i, ix in enumerate(image_ids):
                if ix not in id2findings_t:
                    id2findings_t[ix] = []
                    id2findings_g[ix] = [findings_samples[i]]

                    id2captions_t[ix] = []
                    if len(findings_samples[i]) > 512:
                        findings_samples[i] = findings_samples[i][:511]
                    id2captions_g[ix] = [findings_samples[i]]


                id2findings_t[ix].append(findings_truths[i])
                id2captions_t[ix] = [findings_truths[i]]

                if num_show < 20:
                    print(json.dumps(id2captions_g[ix], ensure_ascii=False))
                    # print(json.dumps(id2captions_t[ix], ensure_ascii=False))
                    num_show += 1

            if count % 100 == 0:
                print(count)
            count += 1

            pred = torch.cat((pred, med_porbs.data), 0)
            gt = torch.cat((gt, medterm_labels), 0)

    print('Total image to be evaluated %d' % (len(id2captions_t)))


    lang_stats = None
    if opt.language_eval == 1:
        lang_stats = eval_utils.evaluate(id2captions_t, id2captions_g, save_to='./results/',
                                         split='test_graph_pretrain')

    return lang_stats


def test():
    ## TODO
    
    tqdm_bar = tqdm(img_test_loader, desc="Testing")
    cnn_model.eval()
    aux_model.eval()
    fusion_model.eval()


    model.eval()

    num_show = 0

    id2findings_t = {}  # true
    id2findings_g = {}  # generated

    id2captions_t = {}
    id2captions_g = {}

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()

    count = 0
    results = {}
    with torch.no_grad():
        for iteration, (ids, image_ids, imgs, input_ids, lm_labels, medterm_labels) in enumerate(tqdm_bar):
            # imgs = [[img.to(device) for img in sample] for sample in imgs]
            imgs = imgs.to(device)
            medterm_labels = medterm_labels.to(device)

            output_global, fm_global, pool_global = cnn_model(imgs)
        
            patchs_var = Attention_gen_patchs(input, fm_global)

            output_local, _, pool_local = aux_model(patchs_var)
            #print(fusion_var.shape)
            output_fusion = fusion_model(pool_global, pool_local)

            med_porbs, findings_seq = model(att_feats=output_fusion, mode='inference', input_type='img')
            findings_samples = decode_transformer_findings(img_train_dataset.idw2word, findings_seq)
            findings_truths = decode_transformer_findings(img_train_dataset.idw2word, lm_labels)
            #print("label", lm_labels)
            #print(lm_labels)
            #print("truth", findings_truths)
            #print("samples", findings_samples)
            sample = {}
            #sample['label'] = lm_labels
            sample['image_id'] = image_ids
            sample['truth'] = findings_truths
            sample['samples'] = findings_samples
            #print(np.size(medterm_labels.cpu().data.numpy()), med_porbs.cpu().data.numpy())
            #sample['medterm_labels'] = list(medterm_labels.cpu().numpy())
            #sample['med_porbs'] = list(med_porbs.cpu().numpy())



            if len(sample['samples'])>512:
                sample['samples'] = sample['samples'][:511]
            results[iteration] = sample

            for i, ix in enumerate(image_ids):
                if ix not in id2findings_t:
                    id2findings_t[ix] = []
                    id2findings_g[ix] = [findings_samples[i]]

                    id2captions_t[ix] = []
                    if len(findings_samples[i]) > 512:
                        findings_samples[i] = findings_samples[i][:511]
                    id2captions_g[ix] = [findings_samples[i]]

                id2findings_t[ix].append(findings_truths[i])
                id2captions_t[ix] = [findings_truths[i]]

                if num_show < 10:
                    print(json.dumps(id2captions_g[ix], ensure_ascii=False))
                    # print(json.dumps(id2captions_t[ix], ensure_ascii=False))
                    num_show += 1

            if count % 100 == 0:
                print(count)
            count += 1

            pred = torch.cat((pred, med_porbs.data), 0)
            gt = torch.cat((gt, medterm_labels), 0)


    print('Total image to be evaluated %d' % (len(id2captions_t)))


    lang_stats = None
    if opt.language_eval == 1:
        lang_stats = eval_utils.evaluate(id2captions_t, id2captions_g, save_to='./results/',
                                         split='test_graph_pretrain')


    AUROCs = compute_AUCs(gt, pred)
    AUROCs = np.array(AUROCs)
    AUROC_avg = AUROCs.mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg), len(AUROCs))

    return lang_stats

def auc_test():
    ## TODO
    
    tqdm_bar = tqdm(img_test_loader, desc="Testing")

    cnn_model.eval()

    model.eval()

    num_show = 0

    id2findings_t = {}  # true
    id2findings_g = {}  # generated

    id2captions_t = {}
    id2captions_g = {}

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()

    count = 0
    with torch.no_grad():
        for iteration, (ids, image_ids, imgs, input_ids, lm_labels, medterm_labels) in enumerate(tqdm_bar):
            # imgs = [[img.to(device) for img in sample] for sample in imgs]
            imgs = imgs.to(device)
            medterm_labels = medterm_labels.to(device)

            if count % 100 == 0:
                print(count)
            count += 1

            pred = torch.cat((pred, medterm_labels), 0)
            gt = torch.cat((gt, medterm_labels), 0)


    lang_stats = None

    AUROCs = compute_AUCs(gt, pred)
    AUROCs = np.array(AUROCs)
    AUROC_avg = AUROCs.mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg), len(AUROCs))

    return lang_stats


def compute_AUCs(gt, pred):
    """
        Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        - gt: Float (n_samples, n_classes), true binary labels
        - pred: Float (n_samples, n_classes),
            can be either probability estimates of the positive class,
            confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    tagdecoder = {}
    cnt = 0
    for i in range(229):
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        except ValueError:
            pass
        tagdecoder[cnt] = i
        cnt += 1
    # print(tagdecoder)
    # with open('./tagdecoder.pkl', 'wb') as f:
    #     pickle.dump(tagdecoder, f) 
        
    return AUROCs
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opt = opts.parse_opt()
    print(opt)
    config = OpenAIGPTConfig('./config/cn_precise_config.json')
    opt.vocab_size = config.vocab_size
    
    train_dataset, train_loader = get_loader2(opt)
    img_train_dataset, img_train_loader = get_loader_cn(opt, 'train')
    img_valid_dataset, img_valid_loader = get_loader_cn(opt, 'val')
    img_test_dataset, img_test_loader = get_loader_cn(opt, 'test')


    with open(config.tag_decoderdir, 'rb') as f:
        tag_decoder = pickle.load(f)


    cnn_model = Densenet121_AG(pretrained=False, num_classes=opt.num_medterm).cuda()
    aux_model = Densenet121_AG(pretrained=False, num_classes =opt.num_medterm).cuda()
    fusion_model = Fusion_Branch(input_size=1024, output_size=opt.num_medterm).cuda()
    model = SentenceLMHeadModel(tag_decoder, config).cuda()
    model = nn.DataParallel(model)
    cnn_model = nn.DataParallel(cnn_model)
    aux_model = nn.DataParallel(aux_model)
    fusion_model = nn.DataParallel(fusion_model)
    
    med_crit = nn.BCELoss().cuda()
    outputs_crit = nn.CrossEntropyLoss(ignore_index=-1).cuda()

    rnn_NoamOpt = NoamOpt(opt.d_model, opt.factor, opt.warmup, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    # rnn_NoamOpt = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay) 
    # cnn_optimizer = optim.Adam(model.parameters(), lr=opt.cnn_learning_rate, weight_decay=opt.weight_decay)
    cnn_optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=opt.weight_decay)
    load_best_model()
    # update_encoder_model(opt.encoder_path)
    # test()
    # eval()
    # auc_test()

    train()
    load_best_model()
    test()
    #auc_test()
    # train()

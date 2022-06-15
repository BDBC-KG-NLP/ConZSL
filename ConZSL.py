from __future__ import print_function
import time
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import util
import classifier
from loss import SupContrastive_Loss, MI_Loss, CrossEntropy_Loss, Margin_Loss
import sys
import model
from knn_classifier import KNearestNeighborsClassifier
from softmax_classifier import SoftmaxClassifier
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='CUB')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=400, help='number features to generate per class')
parser.add_argument('--preprocessing', action='store_true',  default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=312, help='size of semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--latenSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--info_c', type=float, default=0.1, help='information constrain')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=0.2, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--manualSeed', type=int, default=3483,help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--nclass_seen', type=int, default=150, help='number of seen classes')
parser.add_argument('--lr_dec', action='store_true', default=False, help='enable lr decay or not')
parser.add_argument('--lr_dec_ep', type=int, default=1, help='lr decay for every 100 epoch')
parser.add_argument('--lr_dec_rate', type=float, default=0.95, help='lr decay rate')
parser.add_argument('--loss_type', default='contrastive', help='the loss function type')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--use_sin_cons', action='store_true', default=False, help='enable single contrastive loss')
parser.add_argument('--use_tri_cons', action='store_true', default=False, help='enable triplet contrastive loss')
parser.add_argument('--loss_weight', type=float, default=2.0, help='the weight for the loss')
parser.add_argument('--center_margin', type=float, default=190, help='the margin in the center margin loss')
parser.add_argument('--tau', type=float, default=0.5, help='the temperature in contrastive loss')
parser.add_argument('--use_mi', action='store_true', default=False, help='enable minimizing the mutual information')
parser.add_argument('--mi_bound', type=float, default=0.5, help='the bound of mutual infomation')
parser.add_argument('--save_dir', default='./data', help='path to save the model')
parser.add_argument('--test', action='store_true', default=False, help='enable the test mode')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

# initialize generator and discriminator
netG = model.MLP_G(opt)
mapping= model.Mapping(opt)

cls_criterion = nn.NLLLoss()
mi_criterion = MI_Loss(bound=opt.mi_bound)

if opt.loss_type=='contrastive':
    criterion = SupContrastive_Loss(tau=opt.tau)
elif opt.loss_type=='cross_entropy':
    criterion = CrossEntropy_Loss(num_classes=opt.nclass_seen, feat_dim=opt.latenSize, use_gpu=opt.cuda)
elif opt.loss_type=='margin':
    criterion = Margin_Loss(num_classes=opt.nclass_seen, feat_dim=opt.latenSize, margin=opt.center_margin, use_gpu=opt.cuda)
else:
    raise ValueError('Loss function type %s is not supported'%(opt.loss_type))

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
noise1 = torch.FloatTensor(opt.batch_size, opt.nz)
input_label = torch.LongTensor(opt.batch_size)

if opt.cuda:
    mapping.cuda()
    netG.cuda()
    input_res = input_res.cuda()
    noise, noise1, input_att = noise.cuda(), noise1.cuda(), input_att.cuda()
    cls_criterion.cuda()
    criterion.cuda()
    mi_criterion.cuda()
    input_label = input_label.cuda()

def save_model(netG, netD, criterion, syn_data, syn_label):
    torch.save(netG.state_dict(), os.path.join(opt.save_dir, 'netG.pt'))
    torch.save(netD.state_dict(), os.path.join(opt.save_dir, 'netD.pt'))
    torch.save(criterion.state_dict(), os.path.join(opt.save_dir, 'criterion.pt'))
    torch.save(syn_data, os.path.join(opt.save_dir, 'syn_data.pt'))
    torch.save(syn_label, os.path.join(opt.save_dir, 'syn_label.pt'))

def load_model(netG, netD, criterion):
    netG.load_state_dict(torch.load(os.path.join(opt.save_dir, 'netG.pt')))
    netD.load_state_dict(torch.load(os.path.join(opt.save_dir, 'netD.pt')))
    criterion.load_state_dict(torch.load(os.path.join(opt.save_dir, 'criterion.pt')))
    syn_data = torch.load(os.path.join(opt.save_dir, 'syn_data.pt'))
    syn_label = torch.load(os.path.join(opt.save_dir, 'syn_label.pt'))
    return syn_data, syn_label    

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    with torch.no_grad():
        for i in range(nclass):
            iclass = classes[i]
            iclass_att = attribute[iclass]
            syn_att.copy_(iclass_att.repeat(num, 1))
            syn_noise.normal_(0, 1)
            output = netG(syn_noise,syn_att)
            syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
            syn_label.narrow(0, i*num, num).fill_(iclass)
    return syn_feature, syn_label

def plot(X, y, classes, fname):
    # X = TSNE(n_components=2, metric='precomputed').fit_transform(X)
    X = TSNE(n_components=2).fit_transform(X)
    plt.figure()
    # pp = PdfPages('test.pdf')
    colors = ['navy', 'turquoise', 'darkorange', 'blue', 'green', 'magenta', 'yellow', 'lightgreen', 'red', 'black']
    # markers = ['^', '<', '>', 'p', 's', '+', 'X']
    markers = ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.']
    indices = range(len(classes))
    s = 0.3
    for color, i, target_name in zip(colors, indices, classes):
        plt.scatter(X[y == classes[i], 0], X[y == classes[i], 1], color=color, alpha=1.0, s=s, marker=markers[i], 
                    label=target_name)
        # plt.scatter(X[y == i+10, 0], X[y == i+10, 1], color=color, alpha=1.0, s=10., marker=markers[i], 
        #             label=target_name)
    # plt.xlim(-100, 100)
    # plt.ylim(-100, 100)
    plt.legend(loc='best', shadow=True, markerscale=12.0, scatterpoints=1, borderpad=0.1, labelspacing=0.1, handlelength=0.5, handletextpad=0.1, borderaxespad=0.1, columnspacing=0.1, fontsize='large')
    # plt.title('TSNE')
    plt.savefig(fname, format='pdf')

# setup optimizer
optimizerD = optim.Adam(mapping.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
if opt.loss_type!='contrastive':
    optimizer_loss=optim.Adam(criterion.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))

def compute_per_class_acc(test_label, predicted_label, target_classes):
    acc_per_class = 0
    for i in target_classes:
        idx = (test_label == i)
        if torch.sum(idx)==0:
            acc_per_class +=0
        else:
            acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
    acc_per_class /= float(target_classes.size(0))
    return acc_per_class

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)
    _,_,disc_interpolates = netD(interpolates)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

def optimize_beta(beta, MI_loss, alpha=1e-6):
    beta_new = max(0, beta + (alpha * MI_loss))

    # return the updated beta value:
    return beta_new

def train():
    beta=0
    start = time.time()
    best_result_gzsl = {'epoch':0, 'k':5, 'unseen':0., 'seen':0., 'h':0.}
    best_result_gzsl_softmax = {'epoch':0, 'k':5, 'unseen':0., 'seen':0., 'h':0.}

    # pretrain a classifier on seen classes
    pretrain_cls = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses), data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 100)

    for p in pretrain_cls.model.parameters(): # set requires_grad to False
        p.requires_grad = False

    for epoch in range(opt.nepoch):
        mean_lossD = 0
        mean_lossG = 0
        for i in range(0, data.ntrain, opt.batch_size):

            for p in mapping.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            for iter_d in range(opt.critic_iter):
                sample()
                mapping.zero_grad()

                muR,varR,criticD_real = mapping(input_res)

                criticD_real = criticD_real.mean()

                # train with fakeG
                noise.normal_(0, 1)       
                fake = netG(noise, input_att)
                muF, varF, criticD_fake = mapping(fake.detach())

                noise1.normal_(0, 1)
                fake1 = netG(noise1, input_att)
                muF1, varF1, criticD_fake1 = mapping(fake1.detach())

                criticD_fake = criticD_fake.mean()

                # gradient penalty
                gradient_penalty = calc_gradient_penalty(mapping, input_res, fake.data)

                # mutual infomation loss
                if opt.use_mi:
                    mi_loss=mi_criterion(muR, varR)

                # loss
                if opt.loss_type=='margin':
                    super_loss=criterion(muR, input_label)
                elif opt.loss_type=='cross_entropy':
                    super_loss=criterion(muR, input_label)
                elif opt.loss_type=='contrastive':
                    if opt.use_tri_cons:
                        super_loss = criterion(input_label, muR, muF)
                    elif opt.use_sin_cons:
                        super_loss = criterion(input_label, muR)
                    else:
                        super_loss = criterion(input_label, muF, muF1)

                Wasserstein_D = criticD_real - criticD_fake

                D_cost = criticD_fake - criticD_real + gradient_penalty + 0.001*criticD_real**2+super_loss*opt.loss_weight
                if opt.use_mi:
                    D_cost = D_cost + beta*mi_loss

                D_cost.backward()

                optimizerD.step()

                if opt.use_mi:
                    beta=optimize_beta(beta, mi_loss.item())

                if opt.loss_type!='contrastive':
                    optimizer_loss.step()

            # Update G network: optimize WGAN-GP objective

            for p in mapping.parameters(): # reset requires_grad
                p.requires_grad = False # avoid computation

            netG.zero_grad()
            noise.normal_(0, 1)
            fake = netG(noise, input_att)
            _,_,criticG_fake = mapping(fake, train_G=True)
            criticG_fake = criticG_fake.mean()
            G_cost = -criticG_fake

            c_errG_fake = cls_criterion(pretrain_cls.model(fake), input_label)

            errG = G_cost + opt.cls_weight*(c_errG_fake)#+center_loss_f
            errG.backward()
            optimizerG.step()

        if opt.lr_dec:
            if (epoch + 1) % opt.lr_dec_ep == 0:
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = param_group['lr'] * opt.lr_dec_rate
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = param_group['lr'] * opt.lr_dec_rate
                if opt.loss_type!='contrastive':
                    for param_group in optimizer_loss.param_groups:
                        param_group['lr'] = param_group['lr'] * opt.lr_dec_rate                    

        mean_lossG /=  data.ntrain / opt.batch_size
        mean_lossD /=  data.ntrain / opt.batch_size

        end = time.time()

        if opt.use_mi:
            print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, c_errG_fake:%.4f,mi_loss:%.4f,beta:%.4f,super_loss:%.4f,run_time:%d'
                      % (epoch, opt.nepoch, D_cost.item(), G_cost.item(), Wasserstein_D.item(),c_errG_fake.item(),mi_loss.item(),beta,super_loss,end-start))
        else:
            print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, c_errG_fake:%.4f,super_loss:%.4f,run_time:%d'
                      % (epoch, opt.nepoch, D_cost.item(), G_cost.item(), Wasserstein_D.item(),c_errG_fake.item(),super_loss,end-start))

        if epoch % 5 == 0: 

            # evaluate the model, set G to evaluation mode

            netG.eval()
            mapping.eval()

            # Generating fake features for unseen classes
            syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)

            train_X = torch.cat((data.train_feature, syn_feature), 0)
            train_Y = torch.cat((data.train_label, syn_label), 0)

            train_z, _, _ = mapping(train_X.cuda())
            train_y = train_Y.cuda()

            # Testing using a KNN Classifier

            n_neighbors=[1,5,10,15,20,25,30] # search n neighbors
            knn_classifier = KNearestNeighborsClassifier(n_neighbors=n_neighbors)
            test_z_seen, _, _ = mapping(data.test_seen_feature.cuda())
            pred_Y_s = knn_classifier.predict(train_z, train_y, test_z_seen).cpu()
            test_z_unseen, _, _ = mapping(data.test_unseen_feature.cuda())
            pred_Y_u = knn_classifier.predict(train_z, train_y, test_z_unseen).cpu()

            for j, k in enumerate(n_neighbors):
                acc_seen = compute_per_class_acc(pred_Y_s[j], data.test_seen_label, data.seenclasses)
                acc_unseen = compute_per_class_acc(pred_Y_u[j], data.test_unseen_label, data.unseenclasses)
                H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
                print('k=%d unseen=%.4f, seen=%.4f, h=%.4f' % (k, acc_unseen, acc_seen, H))

                if H>best_result_gzsl['h']:
                    best_result_gzsl = {'epoch':epoch, 'k':k, 'unseen':acc_unseen, 'seen':acc_seen, 'h':H}
                    save_model(netG, mapping, criterion, syn_feature, syn_label)

            # Testing using a Softmax Classifier
            sc_gzsl = SoftmaxClassifier(mapping, opt.latenSize, train_X, train_Y, data, opt.nclass_all, opt.cuda,
                                               opt.classifier_lr, 0.5, 25, opt.syn_num)
            acc_unseen, acc_seen, H = sc_gzsl.acc_unseen, sc_gzsl.acc_seen, sc_gzsl.H
            print('unseen=%.4f, seen=%.4f, h=%.4f' % (acc_unseen, acc_seen, H)) 

            if H>best_result_gzsl_softmax['h']:
                best_result_gzsl_softmax = {'epoch':epoch, 'unseen':acc_unseen, 'seen':acc_seen, 'h':H}

            print('Current best GZSL results:', best_result_gzsl)
            print('Current best GZSL softmax results:', best_result_gzsl_softmax)

            netG.train()
            mapping.train()

    print('Current best GZSL results:', best_result_gzsl)
    print('Current best GZSL softmax results:', best_result_gzsl_softmax)


def test():
    syn_feature, syn_label = load_model(netG, mapping, criterion)

    best_result_gzsl = {'k':5, 'unseen':0., 'seen':0., 'h':0.}
    best_result_gzsl_softmax = {'unseen':0., 'seen':0., 'h':0.}

    netG.eval()
    mapping.eval()

    train_X = torch.cat((data.train_feature, syn_feature), 0)
    train_Y = torch.cat((data.train_label, syn_label), 0)

    train_z, _, _ = mapping(train_X.cuda())
    train_y = train_Y.cuda()

    # Testing using a KNN Classifier

    n_neighbors=[1,5,10,15,20,25,30] # search n neighbors
    knn_classifier = KNearestNeighborsClassifier(n_neighbors=n_neighbors)
    test_z_seen, _, _ = mapping(data.test_seen_feature.cuda())
    pred_Y_s = knn_classifier.predict(train_z, train_y, test_z_seen).cpu()
    test_z_unseen, _, _ = mapping(data.test_unseen_feature.cuda())
    pred_Y_u = knn_classifier.predict(train_z, train_y, test_z_unseen).cpu()

    for j, k in enumerate(n_neighbors):
        acc_seen = compute_per_class_acc(pred_Y_s[j], data.test_seen_label, data.seenclasses)
        acc_unseen = compute_per_class_acc(pred_Y_u[j], data.test_unseen_label, data.unseenclasses)
        H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
        print('k=%d unseen=%.4f, seen=%.4f, h=%.4f' % (k, acc_unseen, acc_seen, H))

        if H>best_result_gzsl['h']:
            best_result_gzsl = {'k':k, 'unseen':acc_unseen, 'seen':acc_seen, 'h':H}

    # Testing using a Softmax Classifier
    sc_gzsl = SoftmaxClassifier(mapping, opt.latenSize, train_X, train_Y, data, opt.nclass_all, opt.cuda,
                                       opt.classifier_lr, 0.5, 25, opt.syn_num)
    acc_unseen, acc_seen, H = sc_gzsl.acc_unseen, sc_gzsl.acc_seen, sc_gzsl.H
    print('unseen=%.4f, seen=%.4f, h=%.4f' % (acc_unseen, acc_seen, H)) 

    if H>best_result_gzsl_softmax['h']:
        best_result_gzsl_softmax = {'unseen':acc_unseen, 'seen':acc_seen, 'h':H}

    print('Current best GZSL results:', best_result_gzsl)
    print('Current best GZSL softmax results:', best_result_gzsl_softmax)


if __name__=='__main__':
    if opt.test:
        test()
    else:
        train()



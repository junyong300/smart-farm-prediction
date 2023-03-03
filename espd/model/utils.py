#
# @copyright Electronics and Telecommunications Research Institute (ETRI) All Rights Reserved.
# @author Junyong Park/junyong.park@etri.re.kr, Jong-Ryul Lee, Yong-Hyuk Moon
# @created 2023-01-26
# @last-modified 2023-01-27
#

import os
import math
import logging

import torch
import yaml
from torch import nn
from torch.utils import data
import torch.distributed as dist
from torch.utils.data.sampler import Sampler, BatchSampler
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.metrics import auc
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import torch.nn.functional as F


def read_yaml(str):
    """read yaml files into object"""
    f = open(str, 'r')
    cfg = yaml.safe_load(f)
    subdir_name       = '{}_{}_{}'.format(
        cfg['dataset']['name'],cfg['segmentation'], cfg['backbone'])
    cfg['save-dir']   = os.path.join(cfg['save-dir'], subdir_name)
    print("cfg save-dir:",cfg["save-dir"])
    directory = cfg['save-dir']
    if not os.path.exists(directory):
        os.makedirs(directory)
    return cfg

def config(str):
    """read yaml"""
    f = open(str, 'r')
    cfg = yaml.safe_load(f)

    for n in cfg['combine_dir']:
        print(n)
        cfg[n] = ''.join(cfg['combine_dir'][n])
        os.makedirs(cfg[n], exist_ok = True)

    for n in cfg['combine_path']:
        cfg[n] = ''.join(cfg['combine_path'][n])

    cfg['model_name'] = "{}_exit_{}".format(
        cfg['num_exits'], cfg['model_name'])
    cfg['model_path'] = cfg['model_dir'] + cfg['model_name']

    return cfg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def make_batch_data_sampler(
    sampler, images_per_batch, num_iters=None, start_iter=0):
    """make batch data sampler"""
    batch_sampler = data.sampler.BatchSampler(
        sampler, images_per_batch, drop_last=True)
    if num_iters is not None:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter)
    return batch_sampler

def make_data_sampler(dataset, shuffle, distributed):
    """Make data sampler"""
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = data.sampler.RandomSampler(dataset)
    else:
        sampler = data.sampler.SequentialSampler(dataset)
    return sampler

class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations

class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class FocalLoss(nn.Module):
    """Calculate Focal loss. 
    Used in Segmentation
    """
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class SegmentationMetrics(object):

    def __init__(self, eps=1e-5, average=True, ignore_background=True, activation='0-1'):
        self.eps = eps
        self.average = average
        self.ignore = ignore_background
        self.activation = activation

    @staticmethod
    def _one_hot(gt, pred, class_num):
        # transform sparse mask into one-hot mask
        # shape: (B, H, W) -> (B, C, H, W)
        input_shape = tuple(gt.shape)  # (N, H, W, ...)
        new_shape = (input_shape[0], class_num) + input_shape[1:]
        one_hot = torch.zeros(new_shape).to(pred.device, dtype=torch.float)
        target = one_hot.scatter_(1, gt.unsqueeze(1).long().data, 1.0)
        return target

    @staticmethod
    def _get_class_data(gt_onehot, pred, class_num):
        # perform calculation on a batch
        # for precise result in a single image, plz set batch size to 1
        matrix = np.zeros((3, class_num))

        # calculate tp, fp, fn per class
        for i in range(class_num):
            # pred shape: (N, H, W)
            class_pred = pred[:, i, :, :]
            # gt shape: (N, H, W), binary array where 0 denotes negative and 1 denotes positive
            class_gt = gt_onehot[:, i, :, :]

            pred_flat = class_pred.contiguous().view(-1, )  # shape: (N * H * W, )
            gt_flat = class_gt.contiguous().view(-1, )  # shape: (N * H * W, )

            tp = torch.sum(gt_flat * pred_flat)
            fp = torch.sum(pred_flat) - tp
            fn = torch.sum(gt_flat) - tp

            matrix[:, i] = tp.item(), fp.item(), fn.item()

        return matrix

    def _calculate_multi_metrics(self, gt, pred, class_num):
        # calculate metrics in multi-class segmentation
        matrix = self._get_class_data(gt, pred, class_num)
        if self.ignore:
            matrix = matrix[:, 1:]

        # tp = np.sum(matrix[0, :])
        # fp = np.sum(matrix[1, :])
        # fn = np.sum(matrix[2, :])

        pixel_acc = (np.sum(matrix[0, :]) + self.eps) / (np.sum(matrix[0, :]) + np.sum(matrix[1, :]))
        dice = (2 * matrix[0] + self.eps) / (2 * matrix[0] + matrix[1] + matrix[2] + self.eps)
        precision = (matrix[0] + self.eps) / (matrix[0] + matrix[1] + self.eps)
        recall = (matrix[0] + self.eps) / (matrix[0] + matrix[2] + self.eps)

        if self.average:
            dice = np.average(dice)
            precision = np.average(precision)
            recall = np.average(recall)

        return pixel_acc, dice, precision, recall

    def __call__(self, y_true, y_pred):
        class_num = y_pred.size(1)

        if self.activation in [None, 'none']:
            activation_fn = lambda x: x
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "softmax":
            activation_fn = nn.Softmax(dim=1)
            activated_pred = activation_fn(y_pred)
        elif self.activation == "0-1":
            pred_argmax = torch.argmax(y_pred, dim=1)
            activated_pred = self._one_hot(pred_argmax, y_pred, class_num)
        else:
            raise NotImplementedError("Not a supported activation!")

        gt_onehot = self._one_hot(y_true, y_pred, class_num)
        pixel_acc, dice, precision, recall = self._calculate_multi_metrics(gt_onehot, activated_pred, class_num)
        return pixel_acc, dice, precision, recall

class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.
        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """

        def evaluate_worker(self, pred, label):
            correct, labeled = batch_pix_accuracy(pred, label)
            inter, union = batch_intersection_union(pred, label, self.nclass)

            self.total_correct += correct
            self.total_label += labeled
            if self.total_inter.device != inter.device:
                self.total_inter = self.total_inter.to(inter.device)
                self.total_union = self.total_union.to(union.device)
            self.total_inter += inter
            self.total_union += union

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds, labels):
                evaluate_worker(self, pred, label)

    def get(self):
        """return segmentation result"""
        pixAcc = 1.0 * self.total_correct / (2.22e-16 + self.total_label)
        # remove np.spacing(1)
        IoU = 1.0 * self.total_inter / (2.22e-16 + self.total_union)
        mIoU = IoU.mean().item()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = torch.zeros(self.nclass)
        self.total_union = torch.zeros(self.nclass)
        self.total_correct = 0
        self.total_label = 0

def save_checkpoint(model, cfg):
    """Save Checkpoint"""
    directory = os.path.expanduser(cfg['save-dir'])
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_{}_.pth'.format(
        cfg['dataset']['name'], 
        cfg['segmentation'], cfg['backbone'])
    torch.save(model.state_dict(), os.path.join(directory, filename))

def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    predict = torch.argmax(output.long(), 1) + 1
    target = target.long() + 1

    pixel_labeled = torch.sum(target > 0).item()
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()
    assert pixel_correct <= pixel_labeled, "area should be smaller"
    return pixel_correct, pixel_labeled

def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1

    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()

    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0,\
        "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()

__all__ = ['get_color_pallete', 'print_iou', 'set_img_color',
           'show_prediction', 'show_colorful_images', 'save_colorful_images']

def print_iou(iu, mean_pixel_acc, class_names=None, show_no_back=False):
    """print iou value"""
    n = iu.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = 'Class %d:' % (i + 1)
        else:
            cls = '%d %s' % (i + 1, class_names[i])
        # lines.append('%-8s: %.3f%%' % (cls, iu[i] * 100))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    if show_no_back:
        lines.append('mean_IU: %.3f%% || mean_IU_no_back: %.3f%% \
        || mean_pixel_acc: %.3f%%'
        % (mean_IU * 100, mean_IU_no_back * 100, mean_pixel_acc * 100))
    else:
        lines.append('mean_IU: %.3f%% \
        || mean_pixel_acc: %.3f%%' % (mean_IU * 100, mean_pixel_acc * 100))
    lines.append('=================================================')
    line = "\n".join(lines)
    

def set_img_color(img, label, colors, background=0, show255=False):
    """set image color"""
    for i in range(len(colors)):
        if i != background:
            img[np.where(label == i)] = colors[i]
    if show255:
        img[np.where(label == 255)] = 255

    return img

def show_prediction(img, pred, colors, background=0):
    """show prediction"""
    im = np.array(img, np.uint8)
    set_img_color(im, pred, colors, background)
    out = np.array(im)

    return out

def show_colorful_images(prediction, palettes):
    """show images"""
    im = Image.fromarray(palettes[prediction.astype('uint8').squeeze()])
    im.show()

def save_colorful_images(prediction, filename, output_dir, palettes):
    '''
    :param prediction: [B, H, W, C]
    '''
    im = Image.fromarray(palettes[prediction.astype('uint8').squeeze()])
    fn = os.path.join(output_dir, filename)
    out_dir = os.path.split(fn)[0]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    im.save(fn)


class CELoss(object):
    """calculate crossentropy loss"""
    def compute_bin_boundaries(self, probabilities = np.array([])):
        """compute bin boundaries"""
        #uniform bin spacing
        if probabilities.size == 0:
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]
        else:
            #size of bins 
            bin_n = int(self.n_data/self.n_bins)

            bin_boundaries = np.array([])

            probabilities_sort = np.sort(probabilities)  

            for i in range(0,self.n_bins):
                bin_boundaries = np.append(
                    bin_boundaries,probabilities_sort[i*bin_n])
            bin_boundaries = np.append(bin_boundaries,1.0)

            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]

    def get_probabilities(self, output, labels, logits):
        """get probabilities from softmax"""
        #If not probabilities apply softmax!
        if logits:
            self.probabilities = softmax(output, axis=1)
        else:
            self.probabilities = output

        self.labels = labels
        self.confidences = np.max(self.probabilities, axis=1)
        self.predictions = np.argmax(self.probabilities, axis=1)
        self.accuracies = np.equal(self.predictions,labels)

    def binary_matrices(self):
        """calculate binary matrices"""
        idx = np.arange(self.n_data)
        #make matrices of zeros
        pred_matrix = np.zeros([self.n_data,self.n_class])
        label_matrix = np.zeros([self.n_data,self.n_class])
        #self.acc_matrix = np.zeros([self.n_data,self.n_class])
        pred_matrix[idx,self.predictions] = 1
        label_matrix[idx,self.labels] = 1

        self.acc_matrix = np.equal(pred_matrix, label_matrix)

    def compute_bins(self, index = None):
        """compute bins"""
        self.bin_prop = np.zeros(self.n_bins)
        self.bin_acc = np.zeros(self.n_bins)
        self.bin_conf = np.zeros(self.n_bins)
        self.bin_score = np.zeros(self.n_bins)

        if index == None:
            confidences = self.confidences
            accuracies = self.accuracies
        else:
            confidences = self.probabilities[:,index]
            accuracies = self.acc_matrix[:,index]


        for i, (bin_lower, bin_upper) \
            in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(confidences,bin_lower.item()) * \
                np.less_equal(confidences,bin_upper.item())
            self.bin_prop[i] = np.mean(in_bin)

            if self.bin_prop[i].item() > 0:
                self.bin_acc[i] = np.mean(accuracies[in_bin])
                self.bin_conf[i] = np.mean(confidences[in_bin])
                self.bin_score[i] = np.abs(self.bin_conf[i] - self.bin_acc[i])

class MaxProbCELoss(CELoss):
    """calculate max probability crossentropy loss"""
    def loss(self, output, labels, n_bins = 15, logits = True):
        """calculate loss"""
        self.n_bins = n_bins
        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().compute_bins()

class ConfidenceHistogram(MaxProbCELoss):
    """draw confidence historgram"""
    def plot(self, output, labels, n_bins = 20, logits = True, title = None):
        """plot histogram"""
        super().loss(output, labels, n_bins, logits)
        #scale each datapoint
        n = len(labels)
        w = np.ones(n)/n

        plt.rcParams["font.family"] = "serif"
        #size and axis limits 
        plt.figure(figsize=(3,3))
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xticks(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        plt.yticks(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        #plot grid
        plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)    
        #plot histogram
        plt.hist(
            self.confidences,
            n_bins,
            weights = w,
            color='b',
            range=(0.0,1.0),
            edgecolor = 'k')

        #plot vertical dashed lines
        acc = np.mean(self.accuracies)
        conf = np.mean(self.confidences)                
        plt.axvline(x=acc, color='tab:grey', linestyle='--', linewidth = 3)
        plt.axvline(x=conf, color='tab:grey', linestyle='--', linewidth = 3)
        if acc > conf:
            plt.text(acc+0.03,0.9,'Acc',rotation=90,fontsize=11)
            plt.text(conf-0.07,0.9,'Conf',rotation=90, fontsize=11)
        else:
            plt.text(acc-0.07,0.9,'Acc',rotation=90,fontsize=11)
            plt.text(conf+0.03,0.9,'Conf',rotation=90, fontsize=11)

        plt.ylabel('% of Samples',fontsize=13)
        plt.xlabel('Confidence',fontsize=13)
        plt.tight_layout()
        if title is not None:
            plt.title(title,fontsize=16)
        return plt

class ECELoss(nn.Module):
    """calculate exclusive cross entropy loss"""
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        """calculate loss"""
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(
                bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(
                    avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

class ReliabilityDiagram(MaxProbCELoss):
    """draw reliability diagram"""
    def plot(self, output, labels, n_bins = 10, logits = True, title = None):
        """plot diagram"""
        super().loss(output, labels, n_bins, logits)

        #computations
        delta = 1.0/n_bins
        x = np.arange(0,1,delta)
        mid = np.linspace(delta/2,1-delta/2,n_bins)
        error = np.abs(np.subtract(mid,self.bin_acc))

        plt.rcParams["font.family"] = "serif"
        #size and axis limits
        plt.figure(figsize=(3,3))
        plt.xlim(0,1)
        plt.ylim(0,1)
        #plot grid
        plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)
        #plot bars and identity line
        plt.bar(
            x,
            self.bin_acc,
            color = 'b',
            width=delta,align='edge',edgecolor = 'k',label='Outputs',zorder=5)
        plt.bar(
            x,
            error,
            bottom=np.minimum(self.bin_acc,mid),
            color = 'mistyrose',
            alpha=0.5,
            width=delta,
            align='edge',
            edgecolor = 'r',hatch='/',label='Gap',zorder=10)
        ident = [0.0, 1.0]
        plt.plot(ident,ident,linestyle='--',color='tab:grey',zorder=15)
        #labels and legend
        plt.ylabel('Accuracy',fontsize=13)
        plt.xlabel('Confidence',fontsize=13)
        plt.legend(loc='upper left',framealpha=1.0,fontsize='medium')
        if title is not None:
            plt.title(title,fontsize=16)
        plt.tight_layout()

        return plt

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
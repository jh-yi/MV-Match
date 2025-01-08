import time
from PIL import Image
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from timm.data.auto_augment import auto_augment_transform, rand_augment_transform

import tllib.vision.models as models
from tllib.vision.transforms import ResizeImage
from tllib.utils.metric import accuracy, ConfusionMatrix
from tllib.utils.meter import AverageMeter, ProgressMeter

import datasets as datasets
from datasets.imagelist import MultipleDomainsDataset

def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        # load models from tllib.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    ) + ['Digits']


def get_dataset(dataset_name, root, source, target, train_source_transform, val_transform, train_target_transform=None, method='', sample='', resize_size=224):
    if train_target_transform is None:
        train_target_transform = train_source_transform
    if dataset_name in datasets.__dict__:
        # load datasets from tllib.vision.datasets
        dataset = datasets.__dict__[dataset_name]

        def concat_dataset(tasks, start_idx, **kwargs):
            # return ConcatDataset([dataset(task=task, **kwargs) for task in tasks])
            return MultipleDomainsDataset([dataset(task=task, **kwargs) for task in tasks], tasks, domain_ids=list(range(start_idx, start_idx + len(tasks))))

        train_source_dataset = concat_dataset(root=root, tasks=source, download=True, transform=train_source_transform, start_idx=0, is_train=True, method=method, sample=sample, resize_size=resize_size)
        train_target_dataset = concat_dataset(root=root, tasks=target, download=True, transform=train_target_transform, start_idx=len(source), is_train=True, method=method, sample=sample, resize_size=resize_size)
        val_dataset = concat_dataset(root=root, tasks=target, download=True, transform=val_transform, start_idx=len(source), is_train=False, method=method, sample=sample, resize_size=resize_size)
        if dataset_name=='MiPlo' or dataset_name == 'DomainNet':
            test_dataset = concat_dataset(root=root, tasks=target, split='test', download=True, transform=val_transform, start_idx=len(source), is_train=False, method=method, sample=sample, resize_size=resize_size)
        else:
            test_dataset = val_dataset
        class_names = train_source_dataset.datasets[0].classes
        num_classes = len(class_names)
    else:
        raise NotImplementedError(dataset_name)
    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names


def validate(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[:2]

            if isinstance(images, list):
                if len(images) == 2:
                    images, _ = images
                elif len(images) == 3:
                    images, _, _ = images

            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)

            if isinstance(output, tuple):
                output, f = output

            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        if confmat:
            print(confmat.format(args.class_names))

    return top1.avg


def get_train_transform(resizing='default', scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), random_horizontal_flip=True,
                        random_color_jitter=False, resize_size=224, norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225), auto_augment=None):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    transformed_img_size = 224
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224, scale=scale, ratio=ratio)
        ])
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'ran.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
        transformed_img_size = resize_size
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if auto_augment:
        aa_params = dict(
            translate_const=int(transformed_img_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in norm_mean]),
            interpolation=Image.BILINEAR
        )
        if auto_augment.startswith('rand'):
            transforms.append(rand_augment_transform(auto_augment, aa_params))
        else:
            transforms.append(auto_augment_transform(auto_augment, aa_params))
    elif random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])

class ConfidenceBasedSelfTrainingLoss(nn.Module):
    """
    Self training loss that adopts confidence threshold to select reliable pseudo labels from
    `Pseudo-Label : The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks (ICML 2013)
    <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.664.3543&rep=rep1&type=pdf>`_.

    Args:
        threshold (float): Confidence threshold.
        use_soft_label (bool): Whether use soft labels or hard labels.

    Inputs:
        - y: unnormalized classifier predictions.
        - y_target: unnormalized classifier predictions which will used for generating pseudo labels.

    Returns:
         A tuple, including
            - self_training_loss: self training loss with pseudo labels.
            - mask: binary mask that indicates which samples are retained (whose confidence is above the threshold).
            - pseudo_labels: generated pseudo labels.

    Shape:
        - y, y_target: :math:`(minibatch, C)` where C means the number of classes.
        - self_training_loss: scalar.
        - mask, pseudo_labels :math:`(minibatch, )`.

    """

    def __init__(self, threshold: float, use_soft_label: bool):
        super(ConfidenceBasedSelfTrainingLoss, self).__init__()
        self.threshold = threshold
        self.use_soft_label = use_soft_label

    def forward(self, y, y_target):

        if self.use_soft_label:
            mask = None
            pseudo_labels = None
            self_training_loss = (F.cross_entropy(y, F.softmax(y_target.detach(), dim=1), reduction='none')).mean()
        else:
            confidence, pseudo_labels = F.softmax(y_target.detach(), dim=1).max(dim=1)
            mask = (confidence > self.threshold).float()
            self_training_loss = (F.cross_entropy(y, pseudo_labels, reduction='none') * mask).mean()

        return self_training_loss, mask, pseudo_labels
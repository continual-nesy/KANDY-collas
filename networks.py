import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.resnet import resnet50
from torchvision.models import ResNet50_Weights
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights

from cem.models.cem import ConceptEmbeddingModel

# Implementation of https://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_Decorrelated_Batch_Normalization_CVPR_2018_paper.pdf
# Taken from https://github.com/huangleiBuaa/IterNorm-pytorch
class DBN(nn.Module):
    def __init__(self, num_features, num_groups=32, num_channels=0, dim=4, eps=1e-5, momentum=0.1, affine=True, mode=0,
                 *args, **kwargs):
        super(DBN, self).__init__()
        if num_channels > 0:
            num_groups = num_features // num_channels
        self.num_features = num_features
        self.num_groups = num_groups
        assert self.num_features % self.num_groups == 0
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.mode = mode

        self.shape = [1] * dim
        self.shape[1] = num_features

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(*self.shape))
            self.bias = nn.Parameter(torch.Tensor(*self.shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_groups, 1))
        self.register_buffer('running_projection', torch.eye(num_groups))
        self.reset_parameters()

    # def reset_running_stats(self):
    #     self.running_mean.zero_()
    #     self.running_var.eye_(1)

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor):
        size = input.size()

        assert input.dim() == self.dim and size[1] == self.num_features
        x = input.view(size[0] * size[1] // self.num_groups, self.num_groups, *size[2:])
        training = self.mode > 0 or (self.mode == 0 and self.training)
        x = x.transpose(0, 1).contiguous().view(self.num_groups, -1)
        if training:
            mean = x.mean(1, keepdim=True)
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean
            x_mean = x - mean
            sigma = x_mean.matmul(x_mean.t()) / x.size(1) + self.eps * torch.eye(self.num_groups, device=input.device)
            # print('sigma size {}'.format(sigma.size()))
            u, eig, _ = sigma.svd()
            scale = eig.rsqrt()
            wm = u.matmul(scale.diag()).matmul(u.t())
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * wm
            y = wm.matmul(x_mean)
        else:
            x_mean = x - self.running_mean
            y = self.running_projection.matmul(x_mean)
        output = y.view(self.num_groups, size[0] * size[1] // self.num_groups, *size[2:]).transpose(0, 1)
        output = output.contiguous().view_as(input)
        if self.affine:
            output = output * self.weight + self.bias
        return output

    def extra_repr(self):
        return '{num_features}, num_groups={num_groups}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'mode={mode}'.format(**self.__dict__)

class DBNConceptEmbeddingModel(ConceptEmbeddingModel):
    def __init__(self, decorrelated_bn, num_groups, **kwargs):
        super().__init__(**kwargs)
        self.decorrelated_bn = decorrelated_bn
        if decorrelated_bn:
            self.dbn = DBN(num_features=self.n_concepts, num_groups=num_groups, num_channels=0, dim=2)
        else:
            self.dbn = None

    def _forward(
        self,
        x,
        intervention_idxs=None,
        c=None,
        y=None,
        train=False,
        latent=None,
        competencies=None,
        prev_interventions=None,
        output_embeddings=False,
        output_latent=None,
        output_interventions=None,
    ):
        output_interventions = (
            output_interventions if output_interventions is not None
            else self.output_interventions
        )
        output_latent = (
            output_latent if output_latent is not None
            else self.output_latent
        )
        if latent is None:
            pre_c = self.pre_concept_model(x)
            contexts = []
            c_sem = []

            # First predict all the concept probabilities
            for i, context_gen in enumerate(self.concept_context_generators):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[i]
                context = context_gen(pre_c)
                prob = prob_gen(context)
                contexts.append(torch.unsqueeze(context, dim=1))
                c_sem.append(prob)
            c_sem = torch.cat(c_sem, axis=-1)
            if self.decorrelated_bn:
                c_sem = self.dbn(c_sem)  # Decorrelate logits before sigmoid activation.
            c_sem = self.sig(c_sem)

            contexts = torch.cat(contexts, axis=1)
            latent = contexts, c_sem
        else:
            contexts, c_sem = latent

        # Now include any interventions that we may want to perform!
        if (intervention_idxs is None) and (c is not None) and (
            self.intervention_policy is not None
        ):
            horizon = self.intervention_policy.num_groups_intervened
            if hasattr(self.intervention_policy, "horizon"):
                horizon = self.intervention_policy.horizon
            prior_distribution = self._prior_int_distribution(
                prob=c_sem,
                pos_embeddings=contexts[:, :, :self.emb_size],
                neg_embeddings=contexts[:, :, self.emb_size:],
                competencies=competencies,
                prev_interventions=prev_interventions,
                c=c,
                train=train,
                horizon=horizon,
            )
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                pred_c=c_sem,
                y=y,
                competencies=competencies,
                prev_interventions=prev_interventions,
                prior_distribution=prior_distribution,
            )

        else:
            c_int = c
        if not train:
            intervention_idxs = self._standardize_indices(
                intervention_idxs=intervention_idxs,
                batch_size=x.shape[0],
            )

        # Then, time to do the mixing between the positive and the
        # negative embeddings
        probs, intervention_idxs = self._after_interventions(
            c_sem,
            pos_embeddings=contexts[:, :, :self.emb_size],
            neg_embeddings=contexts[:, :, self.emb_size:],
            intervention_idxs=intervention_idxs,
            c_true=c_int,
            train=train,
            competencies=competencies,
        )
        # Then time to mix!
        c_pred = (
            contexts[:, :, :self.emb_size] * torch.unsqueeze(probs, dim=-1) +
            contexts[:, :, self.emb_size:] * (1 - torch.unsqueeze(probs, dim=-1))
        )
        c_pred = c_pred.view((-1, self.emb_size * self.n_concepts))
        y = self.c2y_model(c_pred)
        tail_results = []
        if output_interventions:
            if (
                (intervention_idxs is not None) and
                isinstance(intervention_idxs, np.ndarray)
            ):
                intervention_idxs = torch.FloatTensor(
                    intervention_idxs
                ).to(x.device)
            tail_results.append(intervention_idxs)
        if output_latent:
            tail_results.append(latent)
        if output_embeddings:
            tail_results.append(contexts[:, :, :self.emb_size])
            tail_results.append(contexts[:, :, self.emb_size:])
        return tuple([c_sem, c_pred, y] + tail_results)


class FakeIdentity(nn.Identity):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features

def generate_backbone(backbone_name: str, input_shape: tuple | list) -> callable:
    def f(output_dim):
        if output_dim is None:
            output_dim = 128 # TODO: parameter instead of being hard-coded. CEM always passes None, CBM changes it.
        c, h, w = input_shape

        # MLP network
        if backbone_name == 'mlp':
            input_size = int(np.prod(input_shape))

            net = nn.Sequential(
                nn.Linear(input_size, 100),
                nn.Tanh(),
                FakeIdentity(100)
            )

        # CNN
        elif backbone_name == 'cnn':

            net = nn.Sequential(
                nn.Conv2d(c, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 128, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveMaxPool2d(output_size=(5, 5)),  # it forces the output resolution to be 5x5
                nn.Flatten(),
                nn.Linear(256 * 5 * 5, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                FakeIdentity(4096)
            )

        # ResNet50 (trained from scratch)
        elif backbone_name == 'resnet50':
            net = resnet50()
            net.fc = FakeIdentity(2048)

        # ResNet50 (pretrained backbone, training head only)
        elif backbone_name == 'resnet50_head_only':

            # ugly hack to fix SSL related issues
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context

            net = resnet50(weights=ResNet50_Weights.DEFAULT)

            for parameter in net.parameters():
                parameter.requires_grad = False

            net.fc = FakeIdentity(2048)

        # ViT Base 16 (pretrained backbone, training head only)
        elif backbone_name == 'vit_head_only':

            # ugly hack to fix SSL related issues
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context

            net = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

            for parameter in net.parameters():
                parameter.requires_grad = False

            net.heads = FakeIdentity(768)

        else:
            raise ValueError(f"Unknown model: {backbone_name}")
        return net

    return f

def generate_net(backbone_name: str, num_outputs: int, input_shape: tuple | list, cem_emb_size: int,
                 n_concepts: int, share_embeddings: bool, decorrelate_probs: bool, num_groups: int) -> \
        tuple[nn.Module, transforms.Compose, transforms.Compose]:
    """Generate a neural network model, moving it to the right device.

        :param backbone_name: Name of the network ('mlp', 'resnet50', ...).
        :param num_outputs: Number of output neurons.
        :param cem_emb_size: Embedding size for a single concept.
        :param n_concepts: Number of concepts in the CEM layer.
        :param input_shape: Shape of the input data (c, h, w).
        :param share_embeddings: Whether to share the linear mapping from c_preds to c_embs.
        :param decorrelate_probs: Whether to add a DecorrelatedBatchNorm layer before c_preds.
        :param num_groups: Number of decorrelation groups for DecorrelatedBatchNorm.
        :returns: The neural network model; the training transforms; the val/test (eval) transforms.
    """

    assert input_shape is not None and isinstance(input_shape, (tuple, list)), \
        "The input_shape field must be associated to a tuple or list with the shape of the input data (e.g. (1,32,32))"
    assert len(input_shape) == 3, \
        "Invalid 'input_shape' options, it must be a tuple like (c, h w)."
    assert num_outputs > 0, \
        "Invalid number of output units, it must be > 0."

    # unpacking
    c, h, w = input_shape

    # transformations (keeping original resolution)
    common_train_transforms = transforms.Compose([
        transforms.RandomRotation(15, fill=127),  # filling with the background color, 127
        transforms.RandomResizedCrop((h, w), scale=(0.95, 1.05), ratio=(0.95, 1.05)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.497, 0.497, 0.497], std=[0.065, 0.065, 0.065]),
    ])

    common_eval_transforms = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.497, 0.497, 0.497], std=[0.065, 0.065, 0.065]),
    ])

    # transformations (changing original resolution to 224x224)
    pretrained_resnet_like_train_transforms = transforms.Compose([
        transforms.RandomRotation(15, fill=127),  # filling with the background color, 127
        transforms.RandomResizedCrop(224, scale=(0.95, 1.05), ratio=(0.95, 1.05)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pretrained_resnet_like_eval_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # MLP network
    if backbone_name == 'mlp':
        train_transforms = common_train_transforms
        train_transforms.transforms.append(torch.flatten)
        eval_transforms = common_eval_transforms
        eval_transforms.transforms.append(torch.flatten)

    # CNN
    elif backbone_name == 'cnn':
        train_transforms = common_train_transforms
        eval_transforms = common_eval_transforms

    # ResNet50 (trained from scratch)
    elif backbone_name == 'resnet50':
        train_transforms = common_train_transforms
        eval_transforms = common_eval_transforms

    # ResNet50 (pretrained backbone, training head only)
    elif backbone_name == 'resnet50_head_only':
        train_transforms = pretrained_resnet_like_train_transforms
        eval_transforms = pretrained_resnet_like_eval_transforms

    # ViT Base 16 (pretrained backbone, training head only)
    elif backbone_name == 'vit_head_only':
        train_transforms = pretrained_resnet_like_train_transforms
        eval_transforms = pretrained_resnet_like_eval_transforms
    else:
        raise ValueError(f"Unknown model: {backbone_name}")

    # discrete_net = DiscreteLatentModel(net, num_outputs, bottleneck_size, discrete_activation)
    #
    # discrete_net.register_buffer("decision_thresholds", 0.5 * torch.ones(num_outputs))
    # return discrete_net, train_transforms, eval_transforms
    cem = DBNConceptEmbeddingModel(
          n_concepts=n_concepts, # Number of training-time concepts
          n_tasks=num_outputs, # Number of output labels
          emb_size=cem_emb_size,
          #concept_loss_weight=0.1,
          #learning_rate=1e-3,
          #optimizer="adam",
          c_extractor_arch=generate_backbone(backbone_name, input_shape), # Replace this appropriately
          training_intervention_prob=0.25, # RandInt probability
          shared_prob_gen=share_embeddings,
          decorrelated_bn=decorrelate_probs,
        num_groups=num_groups
        )

    cem.register_buffer("decision_thresholds", 0.5 * torch.ones(num_outputs))
    return cem, train_transforms, eval_transforms

def save_net(net: tuple[nn.Module] | list[nn.Module] | nn.Module, filename: str) -> None:
    """Save a network to file.

        :param net: A Pytorch net or a list/tuple of nets.
        :param filename: The destination file.
    """

    if not isinstance(net, (tuple, list)):

        # single net
        torch.save(net.state_dict(), filename)
    else:

        # list of nets
        state_dict = [None] * len(net)
        for i in range(0, len(net)):
            state_dict[i] = net[i].state_dict()
        torch.save(state_dict, filename)


def load_net(net: tuple[nn.Module] | list[nn.Module] | nn.Module, filename: str) -> None:
    """Load a network from file.

        :param net: A pre-allocated Pytorch net (or list/tuple of nets), already moved to the right device.
        :param filename: The source file.
    """
    if not isinstance(net, (tuple, list)):

        # single net
        state_dict = torch.load(filename, map_location=net.device)
        net.load_state_dict(state_dict)
    else:

        # list of nets
        state_dict = torch.load(filename, map_location=net[0].device)
        for i in range(0, len(net)):
            net[i].load_state_dict(state_dict[i])

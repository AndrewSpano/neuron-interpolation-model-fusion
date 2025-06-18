import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel, WeightRepresentation


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_batchnorm=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if not use_batchnorm:
            self.bn1 = self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes) if use_batchnorm else nn.Sequential()
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def get_ordered_layers(self):
        layers = [(l, False) for l in [self.conv1, self.conv2]]
        if self.shortcut:
            layers.append((self.shortcut[0], True))
        return layers


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, use_batchnorm=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        if not use_batchnorm:
            self.bn1 = self.bn2 = self.bn3 = nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes) if use_batchnorm else nn.Sequential()
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def get_ordered_layers(self):
        layers = [(l, False) for l in [self.conv1, self.conv2, self.conv3]]
        if self.shortcut:
            layers.append((self.shortcut[0], True))
        return layers


# noinspection PyTypeChecker
class ResNet(BaseModel):
    """Taken from https://github.com/sidak/otfusion/blob/master/cifar.zip,
            which in turn is taken from https://github.com/kuangliu/pytorch-cifar"""

    def __init__(self, name, block, num_blocks, num_classes, use_batchnorm=False, linear_bias=True):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.name = name
        self.block = block
        self.num_blocks = num_blocks
        self.use_batchnorm = use_batchnorm
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if use_batchnorm else nn.Sequential()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes, bias=linear_bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.use_batchnorm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def copy_model(self) -> 'ResNet':
        """Creates a copy of the model with the same architecture,
            and the same weights as the current model."""
        num_classes = self.linear.out_features
        linear_bias = self.linear.bias is not None
        model_copy = ResNet(self.name, self.block, self.num_blocks, num_classes,
                            use_batchnorm=self.use_batchnorm, linear_bias=linear_bias)
        model_copy.load_state_dict(self.state_dict())
        return model_copy

    @property
    def get_ordered_trainable_named_layers(self):
        """Returns the trainable layers of the model in the order they are applied."""
        layers = [WeightRepresentation('conv01', self.conv1.weight, self.conv1)]
        conv_cnv, fc_cnt = 2, 1
        block_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for block_layer in block_layers:
            for block in block_layer:
                for layer, is_shortcut in block.get_ordered_layers():
                    if isinstance(layer, nn.Conv2d):
                        layers.append(WeightRepresentation(f'conv{conv_cnv:02}', layer.weight, layer))
                        conv_cnv += 1
                    elif isinstance(layer, nn.Linear):
                        layers.append(WeightRepresentation(f'fc{fc_cnt:02}', layer.weight, layer))
                        fc_cnt += 1
        layers.append(WeightRepresentation(f'fc{fc_cnt:02}', self.linear.weight, self.linear))
        return layers

    @property
    def get_ordered_trainable_named_levels(self):
        pass


def ResNet18(num_classes, use_batchnorm=True, linear_bias=True):
    return ResNet('ResNet18', BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=num_classes,
                  use_batchnorm=use_batchnorm, linear_bias=linear_bias)


def ResNet34(num_classes, use_batchnorm=True, linear_bias=True):
    return ResNet('ResNet34', BasicBlock, num_blocks=[3, 4, 6, 3], num_classes=num_classes,
                  use_batchnorm=use_batchnorm, linear_bias=linear_bias)


def ResNet50(num_classes, use_batchnorm=True, linear_bias=True):
    return ResNet('ResNet50', Bottleneck, num_blocks=[3, 4, 6, 3], num_classes=num_classes,
                  use_batchnorm=use_batchnorm, linear_bias=linear_bias)


def ResNet101(num_classes, use_batchnorm=True, linear_bias=True):
    return ResNet('ResNet101', Bottleneck, num_blocks=[3, 4, 23, 3], num_classes=num_classes,
                  use_batchnorm=use_batchnorm, linear_bias=linear_bias)


def ResNet152(num_classes, use_batchnorm=True, linear_bias=True):
    return ResNet('ResNet152', Bottleneck, num_blocks=[3, 8, 36, 3], num_classes=num_classes,
                  use_batchnorm=use_batchnorm, linear_bias=linear_bias)

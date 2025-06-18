import torch
import torch.nn as nn

from models.base_model import BaseModel, WeightRepresentation, LevelRepresentation


class VGG(BaseModel):
    """Taken from https://github.com/sidak/otfusion/blob/master/cifar.zip,
        which in turn is taken from https://github.com/kuangliu/pytorch-cifar"""

    cfg = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG11_3/2': [64, 'M', 192, 'M', 384, 384, 'M', 768, 768, 'M', 768, 768, 'M'],
        'VGG11_quad': [64, 'M', 512, 'M', 1024, 1024, 'M', 2048, 2048, 'M', 2048, 512, 'M'],
        'VGG11_doub': [64, 'M', 256, 'M', 512, 512, 'M', 1024, 1024, 'M', 1024, 512, 'M'],
        'VGG11_half': [64, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    }

    def __init__(self, vgg_name, num_classes, input_shape, batch_norm=False, use_bias=True, use_activations=True,
                 train_indices=None, val_indices=None):
        super(VGG, self).__init__(train_indices, val_indices)
        if vgg_name not in self.cfg:
             raise ValueError(f"VGG configuration '{vgg_name}' not found.")

        self.vgg_name = vgg_name
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.use_batch_norm = batch_norm
        self.use_bias = use_bias
        self.use_activations = use_activations

        self.add_relu_to_next_level=False
        self.level_layers = []

        # levels are stored in a ModuleList for seamless access
        self.feature_levels = self._make_levels(VGG.cfg[vgg_name])

        # determine the input size to the classifier dynamically
        with torch.no_grad():
            dummy_input_size = (1, *input_shape)  # ToDo: Adjust for any input size
            dummy_input = torch.randn(dummy_input_size)
            dummy_output = self.forward_features(dummy_input)
            for layer in self.level_layers:
                dummy_output = layer(dummy_output)
            classifier_input_features = dummy_output.flatten(1).shape[1]

        # classifier head
        classifier_layers = self.level_layers + [
            nn.Flatten(start_dim=1),
            nn.Linear(classifier_input_features, num_classes, bias=self.use_bias)
        ]

        self.classifier = nn.Sequential(*classifier_layers)

    def _make_levels(self, cfg):
        levels = nn.ModuleList()
        self.level_layers = []
        in_channels = self.input_shape[0]
        for i, x in enumerate(cfg):

            if self.add_relu_to_next_level:
                self.level_layers.append(nn.ReLU())
                self.add_relu_to_next_level = False

            if x == 'M':
                self.level_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv_layer = nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=self.use_bias)
                self.level_layers.append(conv_layer)
                if self.use_batch_norm:
                    self.level_layers.append(nn.BatchNorm2d(x))

                if self.use_activations:
                    self.level_layers.append(nn.ReLU())
                else:
                    self.add_relu_to_next_level = True

                levels.append(nn.Sequential(*self.level_layers))
                self.level_layers = []
                in_channels = x

        if self.add_relu_to_next_level:
            self.level_layers.append(nn.ReLU())
            self.add_relu_to_next_level = False

        return levels

    def forward_features(self, x):
        for level in self.feature_levels:
            x = level(x)
        return x

    def forward(self, x):
        out = self.forward_features(x)
        out = self.classifier(out)
        return out

    def forward_until_level(self, x: torch.Tensor, level_idx: int = -1) -> torch.Tensor:
        if level_idx == -1:
            return self.forward(x)
        elif level_idx > len(self.get_ordered_trainable_named_levels):
            raise IndexError(f"level_idx {level_idx} is out of bounds. "
                             f"Model has {len(self.get_ordered_trainable_named_levels)} levels (including classifier).")

        # iterate through the features, making sure to include all the levels
        current_level = 0
        for level in self.feature_levels:
            x = level(x)
            if current_level == level_idx:
                return x
            current_level += 1

        # if we reach here, we are at the classifier level
        x = self.classifier(x)
        return x

    def copy_model(self) -> 'VGG':
        """Creates a copy of the model with the same architecture and weights."""
        model_copy = VGG(self.vgg_name, self.num_classes, self.input_shape,
                         batch_norm=self.use_batch_norm, use_bias=self.use_bias, use_activations=self.use_activations)
        model_copy.load_state_dict(self.state_dict())
        return model_copy.to(self.device)

    @property
    def get_ordered_trainable_named_layers(self) -> list[WeightRepresentation]:
        """Returns trainable layers (Conv2d, Linear) in application order."""
        layers = []
        conv_cnt = 0

        # add conv/[bn] layers
        for level in self.feature_levels:
            for layer in level:
                if isinstance(layer, nn.Conv2d):
                    layers.append(WeightRepresentation(f'features.conv{conv_cnt}', layer.weight, layer))
                    conv_cnt += 1
                if self.use_batch_norm and isinstance(layer, nn.BatchNorm2d):
                    layers.append(WeightRepresentation(f'features.bn{conv_cnt}', layer.weight, layer))

        # add the final classifier layer
        classifier = self.classifier[-1]
        layers.append(WeightRepresentation('classifier.fc', classifier.weight, classifier))
        return layers

    @property
    def get_ordered_trainable_named_levels(self) -> list[LevelRepresentation]:
        level_representations = []
        for i, level in enumerate(self.feature_levels):
            conv_layer = level[-1]
            if isinstance(conv_layer, nn.ReLU):
                conv_layer = level[-2]
                if isinstance(conv_layer, nn.BatchNorm2d):
                    conv_layer = level[-3]
            assert isinstance(conv_layer, nn.Conv2d), "For every level, a conv2d layer should be present."

            level_representations.append(LevelRepresentation(
                name=f'level_{i}',
                output_width=conv_layer.out_channels,
                network=level
            ))

        level_representations.append(LevelRepresentation(
            name='classifier',
            output_width=self.num_classes,
            network=self.classifier
        ))

        return level_representations


if __name__ == '__main__':
    # Example usage
    model = VGG('VGG16', num_classes=10, input_shape=(3, 32, 32), batch_norm=False, use_activations=False)
    print(model)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.shape)  # Should be (1, 10)

    breakpoint()

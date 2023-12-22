"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE file.
"""

import torch

def make_normalizer(normalization, num_channels, kwargs={}):
    if normalization == 'bn':
        result = torch.nn.modules.BatchNorm2d(num_channels, **kwargs)
    elif normalization == 'gn':
        if 'num_groups' not in kwargs:
            kwargs['num_groups'] = 8
        result = torch.nn.modules.normalization.GroupNorm(
                num_channels=num_channels,
                **kwargs
                )
    elif normalization is None:
        result = torch.nn.modules.Identity()

    return result

class UNet(torch.nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        depth=5,
        wf=6,
        padding=True,
        normalization=None,
        up_mode='upconv',
        use_skip=True,
        fm_cap=2**10,
        legacy=False,
        double_pool=False

    ):
        """
        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super().__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        self.legacy = legacy
        self.use_skip = use_skip
        self.fm_cap = fm_cap
        self.double_pool = double_pool
        final_out_channels = out_channels
        prev_channels = in_channels

        self.down_path = torch.nn.ModuleList()
        for i in range(depth):
            in_channels = prev_channels
            out_channels = 2 ** (wf + i)
            out_channels = min(self.fm_cap, out_channels)

            #print (i, in_channels, out_channels)
            self.down_path.append(
                UNetConvBlock(in_channels, out_channels, padding, normalization)
            )
            prev_channels = out_channels

        self.up_path = torch.nn.ModuleList()

        #print ("up")
        for i in reversed(range(depth - 1)):
            in_channels = prev_channels

            bridge_channels = 2 ** (wf + i)
            bridge_channels = min(self.fm_cap, bridge_channels)

            out_channels = 2 ** (wf + i)
            out_channels = min(self.fm_cap, out_channels)
            #print (i, in_channels, bridge_channels, out_channels)
            self.up_path.append(
                UNetUpBlock(in_channels, bridge_channels, out_channels, up_mode, padding, normalization, use_skip=self.use_skip,
                    legacy=self.legacy, double_pool=self.double_pool)
            )
            prev_channels = out_channels

        self.last = torch.nn.Conv2d(prev_channels, final_out_channels, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = torch.nn.functional.max_pool2d(x, 2)
                if self.double_pool:
                    x = torch.nn.functional.max_pool2d(x, 2)


        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(torch.nn.Module):
    def __init__(self, in_size, out_size, padding, normalization=None):
        super().__init__()
        block = []

        block.append(torch.nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(torch.nn.ReLU())
        block.append(make_normalizer(normalization, out_size))

        block.append(torch.nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(torch.nn.ReLU())
        block.append(make_normalizer(normalization, out_size))

        self.block = torch.nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(torch.nn.Module):
    def __init__(self, in_size, bridge_size, out_size, up_mode, padding, normalization, use_skip=True, legacy=False, double_pool=False):
        super().__init__()
        self.legacy = legacy


        if double_pool:
            ups_layers = [
                torch.nn.Upsample(mode='bilinear', scale_factor=2),
            ]
            if up_mode == 'upconv':
                ups_layers.append(torch.nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2))
            else:
                ups_layers.append(torch.nn.Upsample(mode='bilinear', scale_factor=2))
                ups_layers.append(torch.nn.Conv2d(in_size, out_size, kernel_size=1))

            self.up = torch.nn.Sequential(*ups_layers)
        else:
            if up_mode == 'upconv':
                self.up = torch.nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
            else:
                self.up = torch.nn.Sequential(
                    torch.nn.Upsample(mode='bilinear', scale_factor=2),
                    torch.nn.Conv2d(in_size, out_size, kernel_size=1),
                )
        self.use_skip = use_skip
        if self.legacy:
            conv_size = in_size
        elif self.use_skip:
            conv_size = bridge_size + out_size
        else:
            conv_size = out_size

        self.conv_block = UNetConvBlock(conv_size, out_size, padding, normalization)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        if self.use_skip:
            crop1 = self.center_crop(bridge, up.shape[2:])
            out = torch.cat([up, crop1], 1)
        else:
            out = up

        out = self.conv_block(out)

        return out

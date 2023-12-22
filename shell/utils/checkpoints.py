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
import os
import urllib
import torch
from torch.utils import model_zoo


class CheckpointIO(object):
    """CheckpointIO class.

    It handles saving and loading checkpoints.

    Args:
        checkpoint_dir (str): path where checkpoints are saved
    """

    def __init__(self, checkpoint_dir="./chkpts", **kwargs):
        self.module_dict = kwargs
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def register_modules(self, **kwargs):
        """Registers modules in current module dictionary."""
        self.module_dict.update(kwargs)

    def save(self, filename, **kwargs):
        """Saves the current module dictionary.

        Args:
            filename (str): name of output file
        """
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        outdict = kwargs
        for k, v in self.module_dict.items():
            outdict[k] = v.state_dict()
        torch.save(outdict, filename)

    def load(self, filename, device="cuda"):
        """Loads a module dictionary from local file or url.

        Args:
            filename (str): name of saved module dictionary
        """
        if is_url(filename):
            return self.load_url(filename)
        else:
            return self.load_file(filename, device)

    def load_file(self, filename, device="cuda"):
        """Loads a module dictionary from file.

        Args:
            filename (str): name of saved module dictionary
        """
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        if os.path.exists(filename):
            state_dict = torch.load(filename, map_location=device)
            scalars = self.parse_state_dict(state_dict)
            return scalars
        else:
            raise FileExistsError

    def load_url(self, url):
        """Load a module dictionary from url.

        Args:
            url (str): url to saved model
        """
        state_dict = model_zoo.load_url(url, progress=True)
        scalars = self.parse_state_dict(state_dict)
        return scalars

    def parse_state_dict(self, state_dict):
        """Parse state_dict of model and return scalars.

        Args:
            state_dict (dict): State dict of model
        """

        for k, v in self.module_dict.items():
            if v is not None:
                if k in state_dict:
                    v.load_state_dict(state_dict[k])
                else:
                    print("Warning: Could not find %s in checkpoint!" % k)
            scalars = {k: v for k, v in state_dict.items() if k not in self.module_dict}
        return scalars


def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ("http", "https")

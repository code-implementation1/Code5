# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import yaml


class Config:
    """
    Configuration namespace. Convert dictionary to members.
    """

    def __init__(self, cfg_list):
        for cfg_dict in cfg_list:
            for k, v in cfg_dict.items():
                setattr(self, k, v)


def parse_yaml(yaml_path, dataset_index):
    """
    Parse the yaml config file.

    Args:
        yaml_path: Path to the yaml config.
        dataset_index: dataset index
    """
    with open(yaml_path, 'r') as fin:
        try:
            cfgs = yaml.load_all(fin.read(), Loader=yaml.FullLoader)
            cfgs = [x for x in cfgs]
            if len(cfgs) == 4:
                if dataset_index == 0:
                    net_param = cfgs[0]
                else:
                    net_param = cfgs[1]
                path_param = cfgs[2]
                dvc_param = cfgs[3]
            else:
                raise ValueError(
                    "At most 3 docs (WHDLD parameters, DLRSD parameters, Path parameters) are supported in config yaml")
        except Exception:
            raise ValueError("Failed to parse yaml")
    return net_param, path_param, dvc_param


def get_config(dataset_index=0):
    """
    Get Config according to the yaml file and cli arguments.
    """
    current_dir = os.path.dirname(__file__)
    yaml_path = os.path.join(current_dir, "../default_config.yaml")
    net_param, path_param, dvc_param = parse_yaml(yaml_path, dataset_index)
    config_list = [net_param, path_param, dvc_param]
    return Config(config_list)

from docQA.errors import ConfigError

import json


class ConfigParser:
    def __init__(self, config_path):
        with open(config_path) as r:
            self.config = json.load(r)
            self.config_path = config_path

    def __setattr__(self, name, value):
        super(ConfigParser, self).__setattr__(name, value)

    def __getattr__(self, name):
        assert name in self.config, ConfigError(name, self.config_path)
        return self.config[name]

    def __getstate__(self):
        return self.__dict__

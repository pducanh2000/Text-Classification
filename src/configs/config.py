import yaml
from yaml.loader import SafeLoader


class Cfg(dict):
    def __init__(self, config_dict=None):
        if config_dict is not None:
            super(Cfg, self).__init__(**config_dict)
        self.__dict__ = self

    @staticmethod
    def load_config_from_file(yaml_path):
        with open(yaml_path, "r") as F:
            config = yaml.load(F, Loader=SafeLoader)
        return Cfg(config)

    def update_config(self, update_dict):
        self.__dict__.update(update_dict)
        return


if __name__ == "__main__":
    test_dict = {
        "a":10,
        "b":10,
    }
    update_dict = {
        "b" : 15,
        "c" : 20
    }
    config = Cfg()
    config = config.load_config_from_file("./config.yaml")
    config.update(update_dict)
    for k, v in config.items():
        print(f"{k} : {v}")

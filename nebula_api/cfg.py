import json
import os
from django.conf import settings


class Cfg:
    """ class to read the config file, section should be a list  """
    def __init__(self, sections, config_name='config.json'):
        path = settings.BASE_DIR + '/../config/' + config_name
        self.file = open(path)
        self._val = {}
        self._config = json.load(self.file)

        for sec in sections:
            self._val[sec] = self._config[sec]

    def __del__(self):
        self.file.close()
        self._val.clear()

    def get(self, section, key, default_val='NA'):
        return self._val.get(section, {}).get(key, default_val)
#        return self._val[section][key]

    def set(self, section, key, val):
        self._val[section][key] = val

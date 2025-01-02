# src/utils/config_handler.py
import json

class Conf:
    def __init__(self, conf_path):
        # 加載配置文件
        with open(conf_path) as f:
            self.conf = json.load(f)
    
    def get(self, key, default=None):
        """獲取配置值"""
        return self.conf.get(key, default)
    
    @property
    def edge_detection(self):
        """獲取邊緣檢測相關的配置"""
        return self.conf.get('edge_detection', {})
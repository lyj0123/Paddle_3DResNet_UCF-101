try:
    from configparser import ConfigParser
except:
    from ConfigParser import ConfigParser

from utils import AttrDict

import logging

logger = logging.getLogger(__name__)

CONFIG_SECS = [
    'train',
    'valid',
    'test',
    'infer',
]

# parse 解析
def parse_config(cfg_file):
    parser = ConfigParser()
    cfg = AttrDict()
    parser.read(cfg_file)
    for sec in parser.sections():
        sec_dict = AttrDict()
        for k, v in parser.items(sec):
            try:
                v = eval(v)         # eval 是一个函数，返回值是参数表达式的值
            except:
                pass
            setattr(sec_dict, k, v) # setattr 函数，用于对已存在的参数进行赋值，或者新建参数并赋值
        setattr(cfg, sec.upper(), sec_dict)

    return cfg

# merge 合并
def merge_configs(cfg, sec, args_dict):
    assert sec in CONFIG_SECS, "invalid config section {}".format(sec)  # CONFIG_SECS 是 train，valid，test，infer 四种模式
    
    sec_dict = getattr(cfg, sec.upper())
    for k, v in args_dict.items():
        if v is None:
            continue
        try:
            if hasattr(sec_dict, k):
                setattr(sec_dict, k, v)
        except:
            pass
    return cfg


def print_configs(cfg, mode):
    logger.info("---------------- {:>5} Arguments ----------------".format( mode))
    for sec, sec_items in cfg.items():
        logger.info("{}:".format(sec))
        for k, v in sec_items.items():
            logger.info("    {}:{}".format(k, v))
    logger.info("-------------------------------------------------\n")

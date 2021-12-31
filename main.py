import sys
from Config import Config
from utils import train_test

if __name__ == '__main__': 
    config = Config()
    '''
    arg = sys.argv
    config.model_name = arg[1]
    config.Limit_Length = int(arg[2])
    '''
    config.dataset_config_init()
    config.model_config_init()
    print(config.__dict__)
    
    train_test(config)
    
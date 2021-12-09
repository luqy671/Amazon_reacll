import sys
from Config import Config
from utils import train_test

if __name__ == '__main__': 
    config = Config()
    config.datasetInfo()
    print(config.__dict__)
    train_test(config)
    
import sys
from Config import Config
from utils import train_test

if __name__ == '__main__':    
    config = Config()
    config.company_cate = sys.argv[1]
    config.datasetInfo()
    config.model_name = config.company_cate + "__GeneNum_test"
    print("Gene_num test in %s"%config.company_cate)
    for Gene_num in [1,2,3,4]:
        config.Gene_num = int(Gene_num)
        print("\n Gene_num = %d"%config.Gene_num)
        print("******************config********************")
        print(config.__dict__)
        train_test(config)
    
class Config():
    def __init__(self):
    # 地址
        self.data_dir = '../data/'
        self.model_save_path = '../model_save/'
        self.ans_save_path = '../ans/'
        self.log_path = '../log/'
        
        #结构/策略 超参
        self.Limit_Length = 1
        
        
        

        # 数据集相关参数
        # Kindle,CDs_and_Vinyl,Electronics,Movies_and_TV
        self.company_cate = 'Kindle' 
        self.test_cut = 20
        self.item_num = 61479

        # 模型相关参数
        self.model_name = 'Intere_Gene_v3'
        self.RNN = 'GRU'

        self.epochs = 100
        self.batch_size = 128
        self.lr = 0.005
        self.L2 = 0
        self.embed_dim = 64
        self.Gene_num = 5
        self.intere_num = 2
        self.sample_num = 10
        self.N_list = [20, 50]
        
    def datasetInfo(self):
        if self.company_cate == 'CDs_and_Vinyl':
            self.item_num = 64304
        if self.company_cate == 'Electronics':
            self.item_num = 62970
        if self.company_cate == 'Movies_and_TV':
            self.item_num = 49930
            

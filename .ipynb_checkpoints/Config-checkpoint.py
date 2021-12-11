class Config():
    def __init__(self):
        
        self.model_name = 'CRF_v3'
        
        # 地址
        self.data_dir = '../data/'
        self.model_save_path = '../model_save/'
        self.ans_save_path = '../ans/'
        self.log_path = '../log/'
        
        #CRF相关
        self.CRF = 0       
        
        #预训练相关
        self.Glove = 0
        self.pre_epochs = 10
        self.Glove_window = 4
        self.co_max = 100
        
        #embed模型限制
        self.Limit_Length = 1
        
        # 数据集相关参数
        # Kindle,CDs_and_Vinyl,Electronics,Movies_and_TV
        self.company_cate = 'Kindle' 
        self.test_cut = 20
        self.item_num = 61479

        # Mulit_Intere模型参数        
        self.Candidate_num = 50
        self.intere_num = 4
        
        
        # 其他       
        self.epochs = 50
        self.batch_size = 128
        self.lr = 0.005
        self.L2 = 0
        self.embed_dim = 64        
        self.sample_num = 10
        self.N_list = [20, 50]
        
    def dataset_config_init(self):
        if self.company_cate == 'CDs_and_Vinyl':
            self.item_num = 64304
        if self.company_cate == 'Electronics':
            self.item_num = 62970
        if self.company_cate == 'Movies_and_TV':
            self.item_num = 49930
            
    def model_config_init(self):
        if self.model_name == 'CRF':
            self.CRF = 1
        
            

import torch
from torch.nn import Module

import os

class Exp:
    def __init__(self):
        # --------------  training config --------------------- #
        self.batch_size = 64
        self.max_epoch = 100
        self.weight_decay = 0
        self.vision_encoder = "ViT-B/32"
        self.basic_lr = 0.0001
        # self.max_lr = 0.0001
        self.betas = (0.9, 0.999)
        self.eps = 1e-8

        # --------------- basic config ----------------- #
        self.data_num_workers = 1
        self.input_size = (224, 224)
        self.test_size = (224, 224)
        self.eval_interval = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.device = torch.device('cuda:1')
        self.save_history_ckpt = True
        self.project_name = "CLIP-fine-tuning-val_loss"
        # torch.backends.cudnn.enabled = False

        self.model = self.get_model(self.vision_encoder)

        # logger.info("GPU MEMORY AVAILABLE: " + str(torch.cuda.mem_get_info()))

        # --------------- dataset path config ----------------- #
        self.output_dir = "/mnt/hdd/volume1/anastasija/CLIP_outputs"
        self.images_path = "/mnt/hdd/volume1/anastasija/CelebA/Img/img_celeba/"
        self.captions_path = "/home/anastasija/Documents/Projects/SBS/CLIP/data/captions_all_attributes.csv"
        self.eval_partitions_path = "/mnt/hdd/volume1/anastasija/CelebA/Eval/"
    
    def get_model(self, vision_encoder):
        """ Get the model for the specified vision encoder and convert it to float or fp32 for faster training """
        import clip
        model, preprocess = clip.load(vision_encoder, device=self.device, jit=False)
        self.model = model
        self.preprocess = preprocess

        return self.model, self.preprocess
    
    # Convert the model
    def convert_models_to_fp32(self, model):
        if self.device == "cpu":
            "Convert the model parameters to float"
            model.float()
        else:
            "Convert the model parameters to fp32"
            for p in model.parameters(): 
                p.data = p.data.float() 
                p.grad.data = p.grad.data.float() 

    
    def get_train_dataset(self):
        from data.dataset import CelebADataset
        train_dataset = CelebADataset(images_path=self.images_path, 
                                      captions_path=self.captions_path, 
                                      eval_partition_path=self.eval_partitions_path, 
                                      preprocess=self.preprocess, 
                                      name="train")
        return train_dataset

    def get_val_dataset(self):
        from data.dataset import CelebADataset
        val_dataset = CelebADataset(images_path=self.images_path, 
                                    captions_path=self.captions_path, 
                                    eval_partition_path=self.eval_partitions_path, 
                                    preprocess=self.preprocess, 
                                    name="val")
        return val_dataset
    
    def get_test_dataset(self):
        from data.dataset import CelebADataset
        test_dataset = CelebADataset(images_path=self.images_path, 
                                     captions_path=self.captions_path, 
                                     eval_partition_path=self.eval_partitions_path, 
                                     preprocess=self.preprocess, 
                                     name="test")
        return test_dataset

    def get_train_dataloader(self, batch_size: int):
        from torch.utils.data import DataLoader
        train_dataset = self.get_train_dataset()
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return self.train_dataloader

    def get_val_dataloader(self, batch_size: int):
        from torch.utils.data import DataLoader
        val_dataset = self.get_val_dataset()
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return self.val_dataloader

    def get_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.basic_lr, 
                                     betas=self.betas, 
                                     eps=self.eps, 
                                     weight_decay=self.weight_decay) # the lr is smaller, more safe for fine tuning to new dataset
        return self.optimizer
    
    def get_lr_scheduler(self):
        import torch.optim.lr_scheduler

        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=self.get_optimizer(), 
                                                           max_lr=self.max_lr, 
                                                           steps_per_epoch=len(self.get_train_dataloader(self.batch_size)), 
                                                           epochs=self.max_epoch)
        
        return self.lr_scheduler

    def get_trainer(self):
        from train_utils.trainer import Trainer

        trainer = Trainer(self)

        return trainer


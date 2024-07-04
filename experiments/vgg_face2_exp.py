import torch
import os

from .base_exp import Exp

class VGGFace2Exp(Exp):
    def __init__(self):
        super().__init__()

        # --------------  training config --------------------- #
        self.batch_size = 64
        self.max_epoch = 100
        self.weight_decay = 0
        self.vision_encoder = "ViT-B/32"
        self.basic_lr = 0.00001
        # self.max_lr = 0.0001
        self.betas = (0.9, 0.999)
        self.eps = 1e-8

        # --------------- basic config ----------------- #
        self.num_workers = 3
        self.input_size = (224, 224)
        self.test_size = (224, 224)
        self.eval_interval = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.device = "cpu"
        self.save_history_ckpt = True
        self.project_name = "CLIP-fine-tuning-VGGFace2"
        # torch.backends.cudnn.enabled = False

        self.model = self.get_model(self.vision_encoder)

        # logger.info("GPU MEMORY AVAILABLE: " + str(torch.cuda.mem_get_info()))

        # --------------- dataset path config ----------------- #
        self.output_dir = "./CLIP_outputs"
        self.vgg2_path = "/ceph/grid/home/am6417/Thesis/Datasets/VGGFace2"
        self.captions_path = "./data/captions/VGGFace2/captions_att_07052024.txt"

    def get_model(self, vision_encoder):
        """ Get the model for the specified vision encoder and convert it to float or fp32 for faster training """
        # from clip import clip_extended
        # import clip
        from clip.clip import load
        model, preprocess = load(vision_encoder, device=self.device, jit=False)
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
                if p.grad is not None:
                    p.data = p.data.float() 
                    p.grad.data = p.grad.data.float() 

    def get_train_dataset(self):
        from data.vgg2_dataset import VGGFace2Dataset
        train_dataset = VGGFace2Dataset(vgg2_path=self.vgg2_path,
                                        captions_path=self.captions_path, 
                                        preprocess=self.preprocess, 
                                        split="train",
                                        num_workers=self.num_workers)
        return train_dataset

    
    def get_val_dataset(self):
        from data.vgg2_dataset import VGGFace2Dataset
        val_dataset = VGGFace2Dataset(vgg2_path=self.vgg2_path,
                                        captions_path=self.captions_path, 
                                        preprocess=self.preprocess,
                                        split="test",
                                        num_workers=self.num_workers)
        return val_dataset
    
    def get_train_dataloader(self, batch_size: int):
        from torch.utils.data import DataLoader
        train_dataset = self.get_train_dataset()
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return self.train_dataloader

    def get_val_dataloader(self, batch_size: int):
        from torch.utils.data import DataLoader
        test_dataset = self.get_val_dataset()
        self.val_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return self.val_dataloader
    
    def get_trainer(self):
        from train_utils.trainer_accelerate import Trainer

        trainer = Trainer(self)

        return trainer
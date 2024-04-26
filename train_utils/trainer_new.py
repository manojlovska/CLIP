import os
import torch
import wandb
from tqdm import tqdm
from statistics import mean
from accelerate import Accelerator, DistributedDataParallelKwargs, get_logger
import clip
from experiments.base_exp import Exp


logger = get_logger(__name__)

class Trainer:
    def __init__(self, exp:Exp):
        self.exp = exp
        self.max_epoch = self.exp.max_epoch
        self.save_history_ckpt = self.exp.save_history_ckpt
        self.accelerator = Accelerator(log_with="wandb", kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
        self.device = self.accelerator.device
        wandb.init(project=self.exp.project_name)

    def train(self):
        self.before_train()
        try:
            self.train_in_epochs()
        except Exception:
            raise
        finally:
            self.after_train()

    def before_train(self):
        logger.info("Experiment details:\n{}".format(self.exp))

        model, _ = self.exp.get_model(self.exp.vision_encoder)

        logger.info("Model Summary: {}".format(model))

        model.to(self.device)

        self.optimizer = self.exp.get_optimizer()
        self.train_dataloader = self.exp.get_train_dataloader(
            batch_size=self.exp.batch_size
            )
        self.val_dataloader = self.exp.get_val_dataloader(
            batch_size=self.exp.batch_size
            )
        self.max_iter = len(self.train_dataloader)

        # Wandb logger
        config = {
            "batch_size": self.exp.batch_size,
            "max_epoch": self.exp.max_epoch,
            "weight_decay": self.exp.weight_decay,
            "vision_encoder": self.exp.vision_encoder,
            "basic_lr": self.exp.basic_lr,
            "betas": self.exp.betas,
            "eps": self.exp.eps,
            "optimizer": self.optimizer
        }
        self.accelerator.init_trackers(self.exp.project_name, 
                                       config=config, 
                                       init_kwargs={"wandb": {"name": wandb.run.name}}
                                       )
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(self.model, 
                                                                                     self.optimizer, 
                                                                                     self.train_dataloader
                                                                                     )
        self.file_name = os.path.join(self.exp.output_dir, self.exp.project_name, wandb.run.name)
        os.makedirs(self.file_name, exist_ok=True)
        logger.info(f"Checkpoints will be saved to: {self.file_name}")

    def train_in_epochs(self):
        for self.epoch in range(self.max_epoch):
            logger.info(f"Training epoch {self.epoch + 1}")
            self.model.train(True)
            pbar = tqdm(self.train_dataloader, total=100) # len(self.train_dataloader)
            for batch in pbar:
                self.optimizer.zero_grad()
                image_names, images, captions, texts = batch 
                logits_per_image, logits_per_text = self.model(images, texts)

                # Calculate the loss
                loss_images = self.contrastive_loss(logits_per_image)
                loss_texts = self.contrastive_loss(logits_per_text)
                self.total_loss = (loss_images + loss_texts) / 2.0
                self.accelerator.backward(self.total_loss)
                
                if self.device != "cpu":
                    self.exp.convert_models_to_fp32(self.model)
                    self.optimizer.step()
                    clip.model.convert_weights(self.model)
                else:
                    self.optimizer.step()
                self.accelerator.log({"train/loss": self.total_loss, "train/learning_rate": self.exp.basic_lr, "train/epoch": self.epoch + 1})
            self.evaluate_and_save_ckpt(self.model)

    def contrastive_loss(self, logits):
        labels = torch.arange(logits.shape[0]).to(self.device)
        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        return loss.mean()

    def after_train(self):
        logger.info("Fine-tuning completed. Best validation loss: {:.2f}".format(self.best_val_loss))
        self.accelerator.end_training()

    def evaluate_and_save_ckpt(self, model):
        v_pbar = tqdm(self.val_dataloader, total=2) # len(self.val_dataloader)

        with torch.no_grad():
            for v_batch in v_pbar:
                val_im_names, val_images, val_captions, val_texts = v_batch
                val_logits_per_image, val_logits_per_text = model(val_images, val_texts)
                val_logits_per_image, val_logits_per_text = self.accelerator.gather_for_metrics(
                    (val_logits_per_image, val_logits_per_text)
                    )
                
                # Validation loss
                val_loss_images = self.contrastive_loss(val_logits_per_image)
                val_loss_texts = self.contrastive_loss(val_logits_per_text)
                self.total_val_loss = (val_loss_images + val_loss_texts) / 2.0
                self.accelerator.log({"val/val_loss": self.total_val_loss})
                self.save_ckpt(f"epoch_{self.epoch + 1}")

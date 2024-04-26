import os
import torch
from experiments.base_exp import Exp
import wandb
# from loguru import logger
from torchsummary import summary
from tqdm.auto import tqdm
import clip
from statistics import mean
import torch.nn.functional as F
import torchvision
import time
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from datetime import timedelta
from accelerate.logging import get_logger

logger = get_logger(__name__)



class Trainer:
    def __init__(self, exp: Exp):
        self.exp = exp

        # training related attr
        self.max_epoch = exp.max_epoch

        # self.device = exp.device
        self.save_history_ckpt = exp.save_history_ckpt

        # data/dataloader related attr
        self.input_size = exp.input_size
        self.max_mean_num_diagonal_max_values_im_percent = 0
        self.best_val_loss = 1e10

        self.accelerator = Accelerator(log_with="wandb", kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True), InitProcessGroupKwargs(timeout=timedelta(seconds=7200))])
        self.device = self.accelerator.device
    
    def train(self):
        self.before_train()
        try:
            self.train_in_epochs()
        except Exception:
            raise
        finally:
            self.after_train()
    
    def before_train(self):
        self.accelerator.print("Exp value:\n{}".format(self.exp))
        # logger.info("Exp value:\n{}".format(self.exp))

        # Model related init
        model, preprocess = self.exp.get_model(self.exp.vision_encoder)

        # logger.info(
        #     "Model Summary: {}".format(model)
        # )

        self.accelerator.print(
            "Model Summary: {}".format(model)
        )

        model.to(self.device)
        
        # Solver related init
        self.optimizer = self.exp.get_optimizer()

        # Data related init
        self.accelerator.print("Loading the training data ...")
        self.train_dataloader = self.exp.get_train_dataloader(
            batch_size=self.exp.batch_size
        )
        self.accelerator.print("Loading the validation data ...")
        self.val_dataloader = self.exp.get_val_dataloader(
            batch_size=self.exp.batch_size
        )

        # Max_iter means iters per epoch
        self.max_iter = len(self.train_dataloader)
        # self.lr_scheduler = self.exp.get_lr_scheduler()
        self.model = model

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
        
        # Accelerator wandb
        if self.accelerator.is_local_main_process:
            wandb.init(project=self.exp.project_name)
            self.accelerator.init_trackers(self.exp.project_name, config=config, init_kwargs={"wandb": {"name": wandb.run.name}})

            # Metric record
            self.file_name = os.path.join(self.exp.output_dir, self.exp.project_name, wandb.run.name)
            os.makedirs(self.file_name, exist_ok=True)

            # logger.info(f"Saving checkpoints into {self.file_name}")
            self.accelerator.print(f"Saving checkpoints into {self.file_name}")

        # Distributed training with accelerator
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader)

        # self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

        # logger.info("Training start...")
        # logger.info("\n{}".format(model))

        self.accelerator.print("Training start...")


    def train_in_epochs(self):
        self.epoch_loss = 0
        self.epoch_val_loss = 0
        self.wandb_step = 0
        self.val_step = 0
        self.train_step = 0
        for self.epoch in range(self.max_epoch):
            # logger.info(f"Start training epoch {self.epoch + 1}")
            self.accelerator.print(f"Start training epoch {self.epoch + 1}")
            self.model.train(True)
            train_iter = 0
            pbar = tqdm(self.train_dataloader, total=len(self.train_dataloader), disable=not self.accelerator.is_local_main_process)
            list_num_diagonal_max_values_im_percent_train = []
            list_num_diagonal_max_values_texts_percent_train = []
            list_mean_max_probs_im = []
            list_mean_max_probs_texts = []
            for batch in pbar:
                self.wandb_step += 1
                self.train_step += 1
                train_iter += 1
                self.optimizer.zero_grad()

                image_names, images, captions, texts = batch 

                # Forward pass
                logits_per_image, logits_per_text = self.model(images, texts)
                
                # Compute loss
                loss_images = self.contrastive_loss(logits_per_image)
                loss_texts = self.contrastive_loss(logits_per_text)

                self.total_loss = (loss_images + loss_texts) / 2.0
                self.epoch_loss += self.total_loss.item()

                # Calculate the metrics for train set
                _, _, num_diagonal_max_values_im_train, mean_max_probs_im = self.get_num_diagonal_max_values(logits_per_image)
                _, _, num_diagonal_max_values_texts_train, mean_max_probs_texts = self.get_num_diagonal_max_values(logits_per_text)


                # Convert validation metrics into percentages
                num_diagonal_max_values_im_percent_train = num_diagonal_max_values_im_train / self.exp.batch_size
                num_diagonal_max_values_texts_percent_train = num_diagonal_max_values_texts_train / self.exp.batch_size

                # Add them to a list to calculate mean value
                list_num_diagonal_max_values_im_percent_train.append(num_diagonal_max_values_im_percent_train.item())
                list_num_diagonal_max_values_texts_percent_train.append(num_diagonal_max_values_texts_percent_train.item())

                list_mean_max_probs_im.append(mean_max_probs_im.item())
                list_mean_max_probs_texts.append(mean_max_probs_texts.item())

                # Backward pass
                self.accelerator.backward(self.total_loss)

                if self.device == "cpu":
                    self.optimizer.step()
                else : 
                    self.exp.convert_models_to_fp32(self.model)
                    self.optimizer.step()
                    clip.model.convert_weights(self.model)
                
                self.accelerator.log({"train/loss": self.total_loss, 
                                       "train/learning_rate": self.exp.basic_lr, 
                                       "train/epoch": self.epoch + 1,
                                       "train_step": self.train_step})

            # Mean values of all batches in epoch
            self.mean_num_diagonal_max_values_im_percent_train = mean(list_num_diagonal_max_values_im_percent_train)
            self.mean_num_diagonal_max_values_texts_percent_train = mean(list_num_diagonal_max_values_texts_percent_train)
            
            self.train_mean_diag_max_prob_im = mean(list_mean_max_probs_im)
            self.train_mean_diag_max_prob_texts = mean(list_mean_max_probs_texts)

            # Epoch train loss is mean of all iterations
            self.epoch_loss = self.epoch_loss / len(pbar)
            # logger.info(f"Epoch {self.epoch+1}/{self.max_epoch}, train loss: {self.epoch_loss}")
            self.accelerator.print(f"Epoch {self.epoch+1}/{self.max_epoch}, train loss: {self.epoch_loss}")

            # Evaluate the model
            if (self.epoch + 1) % 1 == 0:
                if self.accelerator.is_local_main_process:
                    self.num_equal_diagonal_values = self.evaluate_and_save_ckpt(self.model)

                    # Accelerator save the chekcpoint
                    save_checkpoint_accelerate = os.path.join(self.file_name, f"epoch_{self.epoch + 1}")
                    os.makedirs(save_checkpoint_accelerate, exist_ok=True)
                    self.accelerator.save_state(save_checkpoint_accelerate)

    def contrastive_loss(self, logits):
        labels = torch.arange(logits.shape[0]).to(self.device)
        loss = torch.nn.CrossEntropyLoss()(logits, labels)

        return loss.mean()

    def after_train(self):
        # logger.info(
        #     "Fine-tuning of the model is done and the best validation loss is {:.2f}".format(self.best_val_loss)
        # )
        self.accelerator.print(
            "Fine-tuning of the model is done and the best validation loss is {:.2f}".format(self.best_val_loss)
        )

        # self.wandb_logger.finish()
        self.accelerator.end_training()
    
    def evaluate_and_save_ckpt(self, model):
        # Disable gradient computation and reduce memory consumption.
        val_list_num_diagonal_max_values_im_percent = []
        val_list_num_diagonal_max_values_texts_percent = []
        val_list_mean_max_probs_im = []
        val_list_mean_max_probs_texts = []

        with torch.no_grad():
            v_pbar = tqdm(self.val_dataloader, total=len(self.val_dataloader), disable=not self.accelerator.is_local_main_process)
            iter = 0
            model.eval()
            for v_batch in v_pbar:
                self.wandb_step += 1
                self.val_step +=1
                iter = iter+1

                val_im_names, val_images, val_captions, val_texts = v_batch

                val_logits_per_image, val_logits_per_text = model(val_images, val_texts)

                # Compute validation loss
                val_loss_images = self.contrastive_loss(val_logits_per_image)
                val_loss_texts = self.contrastive_loss(val_logits_per_text)

                self.total_val_loss = (val_loss_images + val_loss_texts) / 2.0
                self.epoch_val_loss += self.total_val_loss.item()

                # Log to wandb
                self.accelerator.log({"val/val_loss": self.total_val_loss,
                                       "val_step": self.val_step})
                

                # val_logits_per_image, val_logits_per_text = self.accelerator.gather_for_metrics((val_logits_per_image, val_logits_per_text))

                # Calculate the metrics
                max_values_indices_im, diagonal_max_values_im, num_diagonal_max_values_im, mean_max_probs_im = self.get_num_diagonal_max_values(val_logits_per_image)
                max_values_indices_texts, diagonal_max_values_texts, num_diagonal_max_values_texts, mean_max_probs_texts = self.get_num_diagonal_max_values(val_logits_per_text)

                # Convert validation metrics into percentages
                num_diagonal_max_values_im_percent = num_diagonal_max_values_im / self.val_dataloader.batch_size
                num_diagonal_max_values_texts_percent = num_diagonal_max_values_texts / self.val_dataloader.batch_size

                # Add them to a list to calculate mean value
                val_list_num_diagonal_max_values_im_percent.append(num_diagonal_max_values_im_percent.item())
                val_list_num_diagonal_max_values_texts_percent.append(num_diagonal_max_values_texts_percent.item())

                val_list_mean_max_probs_im.append(mean_max_probs_im.item())
                val_list_mean_max_probs_texts.append(mean_max_probs_texts.item())


                # Save diagonal values tensors for images and texts and equal diagonal values
                save_tensors = {'diagonal_max_values_im': diagonal_max_values_im, 
                                'diagonal_max_values_texts': diagonal_max_values_texts}
                
                filepath = os.path.join(self.file_name, "tensors")
                os.makedirs(filepath, exist_ok=True)
                torch.save(save_tensors, os.path.join(filepath, f"diagonal_max_values_lists_epoch_{self.epoch + 1}_iter_{iter}.pt"))

            # Epoch val loss is mean of all iterations
            self.epoch_val_loss = self.epoch_val_loss / len(v_pbar)
            # logger.info(f"Epoch {self.epoch+1}/{self.max_epoch}, validation loss: {self.epoch_val_loss}")
            self.accelerator.print(f"Epoch {self.epoch+1}/{self.max_epoch}, validation loss: {self.epoch_val_loss}")

            # Mean values of all batches in epoch
            self.mean_num_diagonal_max_values_im_percent = mean(val_list_num_diagonal_max_values_im_percent)
            self.mean_num_diagonal_max_values_texts_percent = mean(val_list_num_diagonal_max_values_texts_percent)
            
            self.val_mean_diag_max_prob_im = mean(val_list_mean_max_probs_im)
            self.val_mean_diag_max_prob_texts = mean(val_list_mean_max_probs_texts)

            update_best_ckpt = self.epoch_val_loss < self.best_val_loss
            self.best_val_loss = min(self.epoch_val_loss, self.best_val_loss)
            self.max_mean_num_diagonal_max_values_im_percent = max(self.max_mean_num_diagonal_max_values_im_percent, self.mean_num_diagonal_max_values_im_percent)

            # Log in logger and wandb
            # logger.info(f"Epoch {self.epoch+1}/{self.max_epoch}, number of maximum diagonal values for image logits: {self.mean_num_diagonal_max_values_im_percent}, number of maximum diagonal values for text logits {self.mean_num_diagonal_max_values_texts_percent}")
            
            self.accelerator.print(f"Epoch {self.epoch+1}/{self.max_epoch}, number of maximum diagonal values for image logits: {self.mean_num_diagonal_max_values_im_percent}, number of maximum diagonal values for text logits {self.mean_num_diagonal_max_values_texts_percent}")

            self.accelerator.log({"val/num_diagonal_max_values_im": self.mean_num_diagonal_max_values_im_percent,
                                    "val/num_diagonal_max_values_texts": self.mean_num_diagonal_max_values_texts_percent,
                                    "val/val_mean_diag_max_prob_im": self.val_mean_diag_max_prob_im,
                                    "val/val_mean_diag_max_prob_texts": self.val_mean_diag_max_prob_texts,
                                    "train/num_diagonal_max_values_im_train": self.mean_num_diagonal_max_values_im_percent_train,
                                    "train/num_diagonal_max_values_texts_train": self.mean_num_diagonal_max_values_texts_percent_train,
                                    "train/mean_diag_max_prob_im": self.train_mean_diag_max_prob_im,
                                    "train/mean_diag_max_prob_texts": self.train_mean_diag_max_prob_texts},
                                    step=self.wandb_step)
            self.wandb_step += 1

            return self.mean_num_diagonal_max_values_im_percent, self.mean_num_diagonal_max_values_texts_percent
   
    
    def get_num_diagonal_max_values(self, logits):
        """ A function to calculate how many times the max logit is in the diagonal for given logit: 
            Inputs: 
                - logits: logits_per_image or logits_per_text 
            Returns:
                - max_value_indices:   tensor of the indices of the maximum values per batch 
                - max_values:          tensor of the maximum values per batch 
                - diagonal_max_values: tensor of the number of times the max value is in the diagonal per batch """
        
        probabilities = logits.softmax(dim=-1) # .cpu().numpy()

        max_values, max_values_indices = probabilities.max(dim=-1)
        diagonal_values = probabilities.diag()

        diagonal_max_values = max_values == diagonal_values

        labels = torch.arange(logits.shape[0]).to(self.device)
        mean_max_probs = max_values[max_values_indices == labels].mean()

        return max_values_indices, diagonal_max_values, diagonal_max_values.sum(), mean_max_probs
    
    def save_ckpt(self, ckpt_name, update_best_ckpt=False, epoch_val_loss=None):
        save_model = self.model
        # logger.info("Save weights to {}".format(self.file_name))
        self.accelerator.print("Save weights to {}".format(self.file_name))

        ckpt_state = {
            "start_epoch": self.epoch + 1,
            "model": save_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "cur_val_loss": epoch_val_loss,
            "cur_train_loss": self.epoch_loss,
            "mean_num_diagonal_max_values_im_percent": self.mean_num_diagonal_max_values_im_percent,
            "mean_num_diagonal_max_values_texts_percent": self.mean_num_diagonal_max_values_texts_percent,
            "mean_diag_max_prob_im": self.val_mean_diag_max_prob_im,
            "mean_diag_max_prob_texts": self.val_mean_diag_max_prob_texts
        }

        self.save_checkpoint(
            ckpt_state,
            update_best_ckpt,
            self.file_name,
            ckpt_name,
        )

    def save_checkpoint(self, state, is_best, save_dir, model_name=""):
        import shutil
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = os.path.join(save_dir, model_name + "_ckpt.pth")
        torch.save(state, filename)
        if is_best:
            best_filename = os.path.join(save_dir, "best_ckpt.pth")
            shutil.copyfile(filename, best_filename)
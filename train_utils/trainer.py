import os
import torch
from experiments.base_exp import Exp
import wandb
from loguru import logger
from torchsummary import summary
from tqdm import tqdm
import clip
from statistics import mean
import torch.nn.functional as F


class Trainer:
    def __init__(self, exp: Exp):
        self.exp = exp

        # training related attr
        self.max_epoch = exp.max_epoch

        self.device = exp.device
        self.save_history_ckpt = exp.save_history_ckpt

        # data/dataloader related attr
        self.input_size = exp.input_size
        self.max_mean_num_equal_diagonal_values_percent = 0
        self.best_val_loss = 1e10

        # metric record
        self.file_name = os.path.join(exp.output_dir, self.exp.project_name)

        # setup_logger(
        #     self.file_name,
        #     distributed_rank=self.rank,
        #     filename="train_log.txt",
        #     mode="a",
        # )
    
    def train(self):
        self.before_train()
        try:
            self.train_in_epochs()
        except Exception:
            raise
        finally:
            self.after_train()
    
    def before_train(self):
        logger.info("Exp value:\n{}".format(self.exp))
        # Model related init
        # torch.cuda.set_device(self.device)
        model, preprocess = self.exp.get_model(self.exp.vision_encoder)

        logger.info(
            "Model Summary: {}".format(model)
        )

        model.to(self.device)
        
        # Solver related init
        self.optimizer = self.exp.get_optimizer()

        # Data related init
        self.train_dataloader = self.exp.get_train_dataloader(
            batch_size=self.exp.batch_size
        )
        self.val_dataloader = self.exp.get_val_dataloader(
            batch_size=self.exp.batch_size
        )

        # Max_iter means iters per epoch
        self.max_iter = len(self.train_dataloader)
        # self.lr_scheduler = self.exp.get_lr_scheduler()
        self.model = model

        # Wandb logger
        self.wandb_logger = wandb.init(project=self.exp.project_name)

        self.wandb_logger.define_metric("train_step")
        self.wandb_logger.define_metric("train/*", step_metric="train_step")

        self.wandb_logger.define_metric("val_step")
        self.wandb_logger.define_metric("val/val_loss", step_metric="val_step")

        # self.wandb_logger.define_metric("metrics_step")

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def train_in_epochs(self):
        self.epoch_loss = 0
        self.epoch_val_loss = 0
        self.wandb_step = 0
        self.val_step = 0
        self.train_step = 0
        for self.epoch in range(self.max_epoch):
            self.model.train(True)
            train_iter = 0
            pbar = tqdm(self.train_dataloader, total=len(self.train_dataloader))
            for batch in pbar:
                self.wandb_step += 1
                self.train_step += 1
                train_iter += 1
                self.optimizer.zero_grad()

                image_names, images, captions, texts = batch 

                images = images.to(self.device)
                texts = texts.to(self.device)

                # Forward pass
                logits_per_image, logits_per_text = self.model(images, texts)
                
                # Compute loss
                loss_images = self.contrastive_loss(logits_per_image)
                loss_texts = self.contrastive_loss(logits_per_text)

                self.total_loss = (loss_images + loss_texts) / 2.0
                self.epoch_loss += self.total_loss.item()

                # Backward pass
                self.total_loss.backward()

                if self.device == "cpu":
                    self.optimizer.step()
                else : 
                    self.exp.convert_models_to_fp32(self.model)
                    self.optimizer.step()
                    clip.model.convert_weights(self.model)
                
                # self.lr_scheduler.step()
                # self.last_lr_in_epoch = self.lr_scheduler.get_last_lr()[0]
                
                self.wandb_logger.log({"train/loss": self.total_loss, 
                                       "train/learning_rate": self.exp.basic_lr, 
                                       "train/epoch": self.epoch + 1,
                                       "train_step": self.train_step})

            # Epoch train loss is mean of all iterations
            self.epoch_loss = self.epoch_loss / len(pbar)
            logger.info(f"Epoch {self.epoch+1}/{self.max_epoch}, train loss: {self.epoch_loss}")

            # Evaluate the model
            if (self.epoch + 1) % 1 == 0:
                self.num_equal_diagonal_values = self.evaluate_and_save_ckpt(self.model)

    def contrastive_loss(self, logits):
        labels = torch.arange(logits.shape[0]).to(self.device)
        loss = torch.nn.CrossEntropyLoss()(logits, labels)

        return loss.mean()

    def after_train(self):
        logger.info(
            "Fine-tuning of the model is done and the best validation loss is {:.2f}".format(self.best_val_loss)
        )
        self.wandb_logger.finish()
    
    def evaluate_and_save_ckpt(self, model):
        # Disable gradient computation and reduce memory consumption.
        list_num_diagonal_max_values_im_percent = []
        list_num_diagonal_max_values_texts_percent = []
        list_num_equal_diagonal_values_percent = []
        list_num_equal_nondiagonal_values_percent = []

        with torch.no_grad():
            iter = 0
            model.eval()
            self.epoch_val_loss = 0
            for vdata in tqdm(self.val_dataloader):
                self.wandb_step += 1
                iter = iter+1
                self.val_step += 1

                val_im_names, val_images, val_captions, val_texts = vdata
                val_images = val_images.to(self.device)
                val_texts = val_texts.to(self.device)

                val_logits_per_image, val_logits_per_text = model(val_images, val_texts)

                # Compute validation loss
                val_loss_images = self.contrastive_loss(val_logits_per_image)
                val_loss_texts = self.contrastive_loss(val_logits_per_text)

                self.total_val_loss = (val_loss_images + val_loss_texts) / 2.0
                self.epoch_val_loss += self.total_val_loss.item()

                # Log to wandb
                self.wandb_logger.log({"val/val_loss": self.total_val_loss,
                                       "val_step": self.val_step})

                # Calculate the metrics
                max_values_indices_im, diagonal_max_values_im, num_diagonal_max_values_im = self.get_num_diagonal_max_values(val_logits_per_image)
                max_values_indices_texts, diagonal_max_values_texts, num_diagonal_max_values_texts = self.get_num_diagonal_max_values(val_logits_per_text)

                # Indicies of maximum probabilities which are in the same places of the diagonal of the images and texts logits
                equal_diagonal_values = diagonal_max_values_im == diagonal_max_values_texts

                # # Transfer the results to cpu, so that the validation is done on cpu
                # equal_diagonal_values = equal_diagonal_values.to(torch.device('cpu'))

                # Number of maximum probabilities which are in the same places of the diagonal of the images and texts logits
                num_equal_diagonal_values = equal_diagonal_values.sum().item()

                # Number of equal nondiagonal values (If the maximum value of images logits and text logits is nondiagonal, check if the indices of the maximums are the same)
                equal_nondiagonal_values = max_values_indices_im[~equal_diagonal_values] == max_values_indices_texts[~equal_diagonal_values]
                num_equal_nondiagonal_values = equal_nondiagonal_values.sum().item()

                # Convert validation metrics into percentages
                num_diagonal_max_values_im_percent = num_diagonal_max_values_im / self.val_dataloader.batch_size
                num_diagonal_max_values_texts_percent = num_diagonal_max_values_texts / self.val_dataloader.batch_size
                num_equal_diagonal_values_percent = num_equal_diagonal_values / self.val_dataloader.batch_size
                num_equal_nondiagonal_values_percent = num_equal_nondiagonal_values / self.val_dataloader.batch_size

                # Add them to a list to calculate mean value
                list_num_diagonal_max_values_im_percent.append(num_diagonal_max_values_im_percent.item())
                list_num_diagonal_max_values_texts_percent.append(num_diagonal_max_values_texts_percent.item())
                list_num_equal_diagonal_values_percent.append(num_equal_diagonal_values_percent)
                list_num_equal_nondiagonal_values_percent.append(num_equal_nondiagonal_values_percent)

                # Save diagonal values tensors for images and texts and equal diagonal values
                save_tensors = {'diagonal_max_values_im': diagonal_max_values_im, 
                                'diagonal_max_values_texts': diagonal_max_values_texts,
                                'equal_diagonal_values': equal_diagonal_values,
                                'num_diagonal_max_values_im': num_diagonal_max_values_im,
                                'num_diagonal_max_values_texts': num_diagonal_max_values_texts,
                                'num_equal_diagonal_values': num_equal_diagonal_values,
                                'num_equal_nondiagonal_values': num_equal_nondiagonal_values}
                
                filepath = os.path.join(self.exp.output_dir, self.exp.project_name, "tensors")
                os.makedirs(filepath, exist_ok=True)
                torch.save(save_tensors, os.path.join(filepath, f"max_values_lists_epoch_{self.epoch + 1}_iter_{iter}.pt"))

            # Epoch val loss is mean of all iterations
            self.epoch_val_loss = self.epoch_val_loss / iter
            logger.info(f"Epoch {self.epoch+1}/{self.max_epoch}, validation loss: {self.epoch_val_loss}")

            # Mean values of all batches in epoch
            mean_num_diagonal_max_values_im_percent = mean(list_num_diagonal_max_values_im_percent)
            mean_num_diagonal_max_values_texts_percent = mean(list_num_diagonal_max_values_texts_percent)
            mean_num_equal_diagonal_values_percent = mean(list_num_equal_diagonal_values_percent)
            mean_num_equal_nondiagonal_values_percent = mean(list_num_equal_nondiagonal_values_percent)

            update_best_ckpt = self.epoch_val_loss < self.best_val_loss
            self.best_val_loss = max(self.epoch_val_loss, self.best_val_loss)
            self.max_mean_num_equal_diagonal_values_percent = max(self.max_mean_num_equal_diagonal_values_percent, mean_num_equal_diagonal_values_percent)

            # Save best checkpoint
            self.save_ckpt("last_epoch", update_best_ckpt, epoch_val_loss=self.epoch_val_loss)
            if self.save_history_ckpt:
                self.save_ckpt(f"epoch_{self.epoch + 1}", epoch_val_loss=self.epoch_val_loss)

            # Log in logger and wandb
            logger.info(f"Epoch {self.epoch+1}/{self.max_epoch}, number of equal diagonal values: {mean_num_equal_diagonal_values_percent}")

            self.wandb_step += 1
            self.wandb_logger.log({"val/num_diagonal_max_values_im": mean_num_diagonal_max_values_im_percent,
                                    "val/num_diagonal_max_values_texts": mean_num_diagonal_max_values_texts_percent,
                                    
                                    "val/num_equal_diagonal_values": mean_num_equal_diagonal_values_percent,
                                    "val/num_equal_nondiagonal_values": mean_num_equal_nondiagonal_values_percent},
                                    step=self.wandb_step)

            return num_equal_diagonal_values_percent

        #         probs = val_logits_per_image.softmax(dim=-1).cpu().numpy()

        #         max_probs_per_img = probs.max(axis=-1)
        #         mean_value = max_probs_per_img.mean()
        #         mean_values_per_batch.append(mean_value)

        # avrg = mean(mean_values_per_batch)

        # update_best_ckpt = avrg > self.best_avrg
        # self.best_avrg = max(self.best_avrg, avrg)

        # self.save_ckpt("last_epoch", update_best_ckpt, avrg=avrg)
        # if self.save_history_ckpt:
        #     self.save_ckpt(f"epoch_{self.epoch + 1}", avrg=avrg)

        # logger.info(f"Epoch  {self.epoch+1}/{self.max_epoch} average score: {avrg}")
        # self.wandb_logger.log({"val/average_probabilities": avrg})

        
    
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

        return max_values_indices, diagonal_max_values, diagonal_max_values.sum()
    
    def save_ckpt(self, ckpt_name, update_best_ckpt=False, epoch_val_loss=None):
        save_model = self.model
        logger.info("Save weights to {}".format(self.file_name))
        ckpt_state = {
            "start_epoch": self.epoch + 1,
            "model": save_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "cur_val_loss": epoch_val_loss,
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
"""
TimesFM Finetuner: A flexible framework for finetuning TimesFM models on custom datasets.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
# from timesfm.pytorch_patched_decoder import create_quantiles

import wandb


class MetricsLogger(ABC):
  """Abstract base class for logging metrics during training.

    This class defines the interface for logging metrics during model training.
    Concrete implementations can log to different backends (e.g., WandB, TensorBoard).
    """

  @abstractmethod
  def log_metrics(self,
                  metrics: Dict[str, Any],
                  step: Optional[int] = None) -> None:
    """Log metrics to the specified backend.

        Args:
          metrics: Dictionary containing metric names and values.
          step: Optional step number or epoch for the metrics.
        """
    pass

  @abstractmethod
  def close(self) -> None:
    """Clean up any resources used by the logger."""
    pass


class WandBLogger(MetricsLogger):
  """Weights & Biases implementation of metrics logging.

    Args:
      project: Name of the W&B project.
      config: Configuration dictionary to log.
      rank: Process rank in distributed training.
    """

  def __init__(self, project: str, config: Dict[str, Any], rank: int = 0):
    self.rank = rank
    if rank == 0:
      wandb.init(project=project, config=config)

  def log_metrics(self,
                  metrics: Dict[str, Any],
                  step: Optional[int] = None) -> None:
    """Log metrics to W&B if on the main process.

        Args:
          metrics: Dictionary of metrics to log.
          step: Current training step or epoch.
        """
    if self.rank == 0:
      wandb.log(metrics, step=step)

  def close(self) -> None:
    """Finish the W&B run if on the main process."""
    if self.rank == 0:
      wandb.finish()


class DistributedManager:
  """Manages distributed training setup and cleanup.

    Args:
      world_size: Total number of processes.
      rank: Process rank.
      master_addr: Address of the master process.
      master_port: Port for distributed communication.
      backend: PyTorch distributed backend to use.
    """

  def __init__(
      self,
      world_size: int,
      rank: int,
      master_addr: str = "localhost",
      master_port: str = "12358",
      backend: str = "nccl",
  ):
    self.world_size = world_size
    self.rank = rank
    self.master_addr = master_addr
    self.master_port = master_port
    self.backend = backend

  def setup(self) -> None:
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = self.master_addr
    os.environ["MASTER_PORT"] = self.master_port

    if not dist.is_initialized():
      dist.init_process_group(backend=self.backend,
                              world_size=self.world_size,
                              rank=self.rank)

  def cleanup(self) -> None:
    """Clean up the distributed environment."""
    if dist.is_initialized():
      dist.destroy_process_group()


@dataclass
class FinetuningConfig:
  """Configuration for model training.

    Args:
      batch_size: Number of samples per batch.
      num_epochs: Number of training epochs.
      learning_rate: Initial learning rate.
      weight_decay: L2 regularization factor.
      freq_type: Frequency, can be [0, 1, 2].
      use_quantile_loss: bool = False  # Flag to enable/disable quantile loss
      quantiles: Optional[List[float]] = None
      device: Device to train on ('cuda' or 'cpu').
      distributed: Whether to use distributed training.
      gpu_ids: List of GPU IDs to use.
      master_port: Port for distributed training.
      master_addr: Address for distributed training.
      use_wandb: Whether to use Weights & Biases logging.
      wandb_project: W&B project name.
      log_every_n_steps: Log metrics every N steps (batches), this is inspired from Pytorch Lightning
      val_check_interval: How often within one training epoch to check val metrics. (also from Pytorch Lightning)
        Can be: float (0.0-1.0): fraction of epoch (e.g., 0.5 = validate twice per epoch)
                int: validate every N batches
    """

  batch_size: int = 32
  num_epochs: int = 20
  learning_rate: float = 1e-4
  weight_decay: float = 0.01
  freq_type: int = 0
  use_quantile_loss: bool = False
  quantiles: Optional[List[float]] = None
  device: str = "cuda" if torch.cuda.is_available() else "cpu"
  distributed: bool = False
  gpu_ids: List[int] = field(default_factory=lambda: [0])
  master_port: str = "12358"
  master_addr: str = "localhost"
  use_wandb: bool = False
  wandb_project: str = "timesfm-finetuning"
  log_every_n_steps: int = 50
  val_check_interval: float = 0.5


class TimesFMFinetuner:
  """Handles model training and validation.

    Args:
      model: PyTorch model to train.
      config: Training configuration.
      rank: Process rank for distributed training.
      loss_fn: Loss function (defaults to MSE).
      logger: Optional logging.Logger instance.
    """

  def __init__(
      self,
      model: nn.Module,
      MMTimesFM_model: nn.Module,
      config: FinetuningConfig,
      rank: int = 0,
      loss_fn: Optional[Callable] = None,
      logger: Optional[logging.Logger] = None,
  ):
    self.model = model
    self.MMTimesFM_model = MMTimesFM_model
    self.config = config
    self.rank = rank
    self.logger = logger or logging.getLogger(__name__)
    self.device = torch.device(
        f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    self.loss_fn = loss_fn or (lambda x, y: torch.mean((x - y.squeeze(-1))**2))

    if config.use_wandb:
      self.metrics_logger = WandBLogger(config.wandb_project, config.__dict__,
                                        rank)

    if config.distributed:
      self.dist_manager = DistributedManager(
          world_size=len(config.gpu_ids),
          rank=rank,
          master_addr=config.master_addr,
          master_port=config.master_port,
      )
      self.dist_manager.setup()
      self.model = self._setup_distributed_model()

  def _setup_distributed_model(self) -> nn.Module:
    """Configure model for distributed training."""
    self.model = self.model.to(self.device)
    return DDP(self.model,
               device_ids=[self.config.gpu_ids[self.rank]],
               output_device=self.config.gpu_ids[self.rank])

  def _create_dataloader(self, dataset: Dataset, is_train: bool) -> DataLoader:
    """Create appropriate DataLoader based on training configuration.

        Args:
          dataset: Dataset to create loader for.
          is_train: Whether this is for training (affects shuffling).

        Returns:
          DataLoader instance.
        """
    if self.config.distributed:
      sampler = torch.utils.data.distributed.DistributedSampler(
          dataset,
          num_replicas=len(self.config.gpu_ids),
          rank=dist.get_rank(),
          shuffle=is_train)
    else:
      sampler = None

    def custom_collate_fn(batch):
      # print(f"Batch shapes: {[x.shape for x in batch]}")  # 打印每个张量的形状
      # print(f"Batch: {batch}")
      x_context = torch.stack([item[0] for item in batch])
      input_padding = torch.stack([item[1] for item in batch])
      freq = torch.stack([item[2] for item in batch])
      x_future = torch.stack([item[3] for item in batch])
      return x_context, input_padding, freq, x_future
  
    return DataLoader(
        dataset,
        batch_size=self.config.batch_size,
        shuffle=(is_train and not self.config.distributed),
        sampler=sampler,
        collate_fn=custom_collate_fn,
    )

  def _quantile_loss(self, pred: torch.Tensor, actual: torch.Tensor,
                     quantile: float) -> torch.Tensor:
    """Calculates quantile loss.
        Args:
            pred: Predicted values
            actual: Actual values
            quantile: Quantile at which loss is computed
        Returns:
            Quantile loss
        """
    dev = actual - pred
    loss_first = dev * quantile
    loss_second = -dev * (1.0 - quantile)
    return 2 * torch.where(loss_first >= 0, loss_first, loss_second)

  # def _process_batch(self, batch: List[torch.Tensor]) -> tuple:
  #   """Process a single batch of data.

  #       Args:
  #         batch: List of input tensors.

  #       Returns:
  #         Tuple of (loss, predictions).
  #       """
  #   x_context, x_padding, freq, x_future = [
  #       t.to(self.device, non_blocking=True) for t in batch
  #   ]

  #   predictions = self.model(x_context, x_padding.float(), freq)
  #   predictions_mean = predictions[..., 0]
  #   last_patch_pred = predictions_mean[:, -1, :]

  #   loss = self.loss_fn(last_patch_pred, x_future.squeeze(-1))
  #   if self.config.use_quantile_loss:
  #     quantiles = self.config.quantiles or create_quantiles()
  #     for i, quantile in enumerate(quantiles):
  #       last_patch_quantile = predictions[:, -1, :, i + 1]
  #       loss += torch.mean(
  #           self._quantile_loss(last_patch_quantile, x_future.squeeze(-1),
  #                               quantile))

  #   return loss, predictions

  def _process_batch(self, batch: List[torch.Tensor], texts: List[Optional[List[str]]]) -> tuple:
    """Process a batch for training MMTimesFM.

    Args:
        batch: List of input tensors.
        texts: Corresponding text descriptions.

    Returns:
        Tuple of (loss, predictions).
    """
    x_context, x_padding, freq, x_future = [t.to(self.device, non_blocking=True) for t in batch]
    

    # **Step 1: 使用 TimesFM 预测 x_{t+1}**
    with torch.no_grad():
        timesfm_pred = self.model(x_context, x_padding.float(), freq)
        timesfm_pred_mean = timesfm_pred[..., 0]
        x_t1_hat = timesfm_pred_mean[:, -1, :]  # 取最后时间步的预测值
        print(f"x_t1_hat: {x_t1_hat}")
    # **Step 2: 计算 TimesFM 误差 δ(x_{t+1})**
    delta_x_t1 = x_future.squeeze(-1) - x_t1_hat  # 真实值 - 预测值
    print(f"Delta x_t1: {delta_x_t1}")

    # **Step 3: 让 MMTimesFM 预测该误差**
    if texts is not None:
        # texts = texts.to(self.device, non_blocking=True)
        print(f"Texts: {texts}")
    mm_timesfm_pred = self.MMTimesFM_model.forward(x_context, x_padding.float(), freq, texts)
    mm_timesfm_pred_mean = mm_timesfm_pred[..., 0]
    delta_x_t1_hat = mm_timesfm_pred_mean[:, -1, :]  # MMTimesFM 预测误差
    print(f"Delta x_t1_hat: {delta_x_t1_hat}")

    # **Step 4: 计算 MSE 损失**
    loss = self.loss_fn(delta_x_t1_hat, delta_x_t1)
    print(f"Loss: {loss}")

    return loss, mm_timesfm_pred
  
  def _train_epoch(self, train_loader: DataLoader, train_texts: List[str],
                   optimizer: torch.optim.Optimizer) -> float:
    """Train for one epoch in a distributed setting.

        Args:
            train_loader: DataLoader for training data.
            train_texts: List of text descriptions.
            optimizer: Optimizer instance.

        Returns:
            Average training loss for the epoch.
        """
    self.MMTimesFM_model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    # print("train_texts: ",train_texts)
    # print("train_texts len: ",len(train_texts))

    train_texts = [t if t is not None else [] for t in train_texts]
    while len(train_texts) < len(train_loader):
      train_texts.append([])  # 用空文本填充
    text_batch_start = 0
    for i, (batch) in enumerate(train_loader):
      text_batch = train_texts[text_batch_start:text_batch_start + len(batch)] 
      text_batch_start += len(batch)
      print(f"[DEBUG] Batch {i}: batch size = {len(batch)}, text batch size = {len(text_batch)}")
      print(f"[DEBUG] Batch {i}: batch = {batch}, text batch = {text_batch}")
      if train_texts:
        loss, _ = self._process_batch(batch, text_batch)
      else:
        loss, _ = self._process_batch(batch)

      optimizer.zero_grad()
      loss.backward()

      # Apply gradient clipping before the optimizer step
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
      optimizer.step()

      total_loss += loss.item()

    avg_loss = total_loss / num_batches

    if self.config.distributed:
      avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
      dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
      avg_loss = (avg_loss_tensor / dist.get_world_size()).item()

    return avg_loss

  def _validate(self, val_loader: DataLoader, val_texts: List[str],) -> float:
    """Perform validation.

        Args:
            val_loader: DataLoader for validation data.
            val_texts: List of text descriptions.

        Returns:
            Average validation loss.
        """
    self.model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)

    with torch.no_grad():
      for i, batch in enumerate(val_loader):
        if val_texts:
          loss, _ = self._process_batch(batch, val_texts[i])
        else:
          loss, _ = self._process_batch(batch)
        total_loss += loss.item()

    avg_loss = total_loss / num_batches

    if self.config.distributed:
      avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
      dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
      avg_loss = (avg_loss_tensor / dist.get_world_size()).item()

    return avg_loss
  
  def _load_checkpoint(self, ckpt_path: str) -> None:
        """Load the pre - trained checkpoint of TimesFM.

        Args:
            ckpt_path (str): The path to the checkpoint file of the pre - trained model.
        """
        if os.path.exists(ckpt_path):
            # Load the checkpoint from the specified path to the given device
            checkpoint = torch.load(ckpt_path, map_location=self.device)

            # Compatibility for weights trained with DataParallel or DDP
            if "module." in list(checkpoint.keys())[0]:
                # Remove the "module." prefix from the keys in the checkpoint
                checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

            # Load the state dictionary into the model
            self.model.load_state_dict(checkpoint, strict=False)
            self.logger.info(f"Successfully loaded checkpoint: {ckpt_path}")
        else:
            self.logger.warning(f"Checkpoint {ckpt_path} does not exist, skipping loading.")

  def finetune(self, train_dataset: Dataset,
               val_dataset: Dataset, train_texts: Optional[List[str]] = None, val_texts: Optional[List[str]] = None, ckpt_path: Optional[str] = None) -> Dict[str, Any]:
    """Train the model.

        Args:
          train_dataset: Training dataset.
          val_dataset: Validation dataset.
          train_texts: List of training texts.
          val_texts: List of validation texts.
          ckpt_path: Path to a pre-trained checkpoint.

        Returns:
          Dictionary containing training history.
        """
    self.model = self.model.to(self.device)
    self.MMTimesFM_model = self.MMTimesFM_model.to(self.device)
    # # **加载 checkpoint**
    # if ckpt_path:
    #     self._load_checkpoint(ckpt_path)
    train_loader = self._create_dataloader(train_dataset, is_train=True)
    val_loader = self._create_dataloader(val_dataset, is_train=False)

    # optimizer = torch.optim.Adam(self.model.parameters(),
    #                              lr=self.config.learning_rate,
    #                              weight_decay=self.config.weight_decay)
    optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=0.9)

    history = {"train_loss": [], "val_loss": [], "learning_rate": []}

    self.logger.info(
        f"Starting training for {self.config.num_epochs} epochs...")
    self.logger.info(f"Training samples: {len(train_dataset)}")
    self.logger.info(f"Validation samples: {len(val_dataset)}")

    try:
      for epoch in range(self.config.num_epochs):
        train_loss = self._train_epoch(train_loader, train_texts, optimizer)
        val_loss = self._validate(val_loader, val_texts)
        current_lr = optimizer.param_groups[0]["lr"]

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
            "epoch": epoch + 1,
        }

        if self.config.use_wandb:
          self.metrics_logger.log_metrics(metrics)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)

        if self.rank == 0:
          self.logger.info(
              f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
          )

    except KeyboardInterrupt:
      self.logger.info("Training interrupted by user")

    if self.config.distributed:
      self.dist_manager.cleanup()

    if self.config.use_wandb:
      self.metrics_logger.close()

    return {"history": history}

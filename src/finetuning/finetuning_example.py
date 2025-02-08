"""
Example usage of the TimesFM Finetuning Framework.

For single GPU:
python script.py --training_mode=single

For multiple GPUs:
python script.py --training_mode=multi --gpu_ids=0,1,2
"""

import os
from os import path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import yfinance as yf
from absl import app, flags
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset

from finetuning.finetuning_torch_modify import FinetuningConfig, TimesFMFinetuner
from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder
import MMtimesFM

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "training_mode",
    "single",
    ["single", "multi"],
    'Training mode: "single" for single-GPU or "multi" for multi-GPU training.',
)

flags.DEFINE_list(
    "gpu_ids", ["0"],
    "Comma-separated list of GPU IDs to use for multi-GPU training. Example: 0,1,2"
)


class TimeSeriesDataset(Dataset):
  """Dataset for time series data compatible with TimesFM."""

  def __init__(self,
               series: np.ndarray,
               context_length: int,
               horizon_length: int,
               freq_type: int = 0):
    """
        Initialize dataset.

        Args:
            series: Time series data
            context_length: Number of past timesteps to use as input
            horizon_length: Number of future timesteps to predict
            freq_type: Frequency type (0, 1, or 2)
        """
    if freq_type not in [0, 1, 2]:
      raise ValueError("freq_type must be 0, 1, or 2")

    self.series = series
    self.context_length = context_length
    self.horizon_length = horizon_length
    self.freq_type = freq_type
    self._prepare_samples()

  def _prepare_samples(self) -> None:
    """Prepare sliding window samples from the time series."""
    self.samples = []
    total_length = self.context_length + self.horizon_length

    for start_idx in range(0, len(self.series) - total_length + 1):
      end_idx = start_idx + self.context_length
      x_context = self.series[start_idx:end_idx]
      x_future = self.series[end_idx:end_idx + self.horizon_length]
      self.samples.append((x_context, x_future))

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(
      self, index: int
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_context, x_future = self.samples[index]

    x_context = torch.tensor(x_context, dtype=torch.float32)
    x_future = torch.tensor(x_future, dtype=torch.float32)

    input_padding = torch.zeros_like(x_context)
    freq = torch.tensor([self.freq_type], dtype=torch.long)

    return x_context, input_padding, freq, x_future


def prepare_datasets(series: np.ndarray, texts: list, context_length: int, horizon_length: int, freq_type: int = 0, train_split: float = 0.8):
    """
    Prepare datasets for multiple time series (each news article has a time series).

    Args:
        series: Time series data (shape: [num_samples, num_timesteps])
        texts: Corresponding text data (list of strings)
        context_length: Number of past timesteps to use
        horizon_length: Number of future timesteps to predict
        freq_type: Frequency type (0, 1, or 2)
        train_split: Fraction of data to use for training

    Returns:
        Tuple of (train_dataset, val_dataset, train_texts, val_texts)
    """
    num_samples = series.shape[0]  # 文章数量
    train_size = int(num_samples * train_split)

    train_data, val_data = series[:train_size], series[train_size:]
    train_texts, val_texts = texts[:train_size], texts[train_size:]

    train_dataset = [TimeSeriesDataset(ts, context_length, horizon_length, freq_type) for ts in train_data]
    val_dataset = [TimeSeriesDataset(ts, context_length, horizon_length, freq_type) for ts in val_data]

    return train_dataset, val_dataset, train_texts, val_texts


def get_models(load_timesfm_weights: bool = False):
    """Load TimesFM (pretrained) and initialize MMTimesFM."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # repo_id = "google/timesfm-2.0-500m-pytorch"
    checkpoint_path="/home/zzx/projects/rrg-timsbc/zzx/torch_model.ckpt"

    # 加载 TimesFM
    hparams = TimesFmHparams(
        backend=device,
        per_core_batch_size=32,
        horizon_len=128,
        num_layers=50,
        use_positional_embedding=False,
        context_len=192,
    )
    timesfm = TimesFm(hparams=hparams, checkpoint=TimesFmCheckpoint(path=path))

    timesfm_model = PatchedTimeSeriesDecoder(timesfm._model_config)
    if load_timesfm_weights:
        # checkpoint_path = path.join(snapshot_download(repo_id), "torch_model.ckpt")
        loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
        timesfm_model.load_state_dict(loaded_checkpoint)
    timesfm_model.eval()  # TimesFM 不训练，只做推理

    # 初始化 MMTimesFM（用户训练）
    mm_timesfm_model = MMtimesFM()  # MMTimesFM 结构类似 TimesFM
    mm_timesfm_model.train()  # MMTimesFM 需要训练

    return timesfm_model, mm_timesfm_model, hparams, timesfm._model_config


def plot_predictions(
    model: TimesFm,
    val_dataset: Dataset,
    save_path: Optional[str] = "predictions.png",
) -> None:
  """
    Plot model predictions against ground truth for a batch of validation data.

    Args:
      model: Trained TimesFM model
      val_dataset: Validation dataset
      save_path: Path to save the plot
    """
  import matplotlib.pyplot as plt

  model.eval()

  x_context, x_padding, freq, x_future = val_dataset[0]
  x_context = x_context.unsqueeze(0)  # Add batch dimension
  x_padding = x_padding.unsqueeze(0)
  freq = freq.unsqueeze(0)
  x_future = x_future.unsqueeze(0)

  device = next(model.parameters()).device
  x_context = x_context.to(device)
  x_padding = x_padding.to(device)
  freq = freq.to(device)
  x_future = x_future.to(device)

  with torch.no_grad():
    predictions = model(x_context, x_padding.float(), freq)
    predictions_mean = predictions[..., 0]  # [B, N, horizon_len]
    last_patch_pred = predictions_mean[:, -1, :]  # [B, horizon_len]

  context_vals = x_context[0].cpu().numpy()
  future_vals = x_future[0].cpu().numpy()
  pred_vals = last_patch_pred[0].cpu().numpy()

  context_len = len(context_vals)
  horizon_len = len(future_vals)

  plt.figure(figsize=(12, 6))

  plt.plot(range(context_len),
           context_vals,
           label="Historical Data",
           color="blue",
           linewidth=2)

  plt.plot(
      range(context_len, context_len + horizon_len),
      future_vals,
      label="Ground Truth",
      color="green",
      linestyle="--",
      linewidth=2,
  )

  plt.plot(range(context_len, context_len + horizon_len),
           pred_vals,
           label="Prediction",
           color="red",
           linewidth=2)

  plt.xlabel("Time Step")
  plt.ylabel("Value")
  plt.title("TimesFM Predictions vs Ground Truth")
  plt.legend()
  plt.grid(True)

  if save_path:
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

  plt.close()


def get_data(context_len: int,
             horizon_len: int,
             freq_type: int = 0) -> Tuple[Dataset, Dataset]:
  df = yf.download("AAPL", start="2010-01-01", end="2019-01-01")
  time_series = df["Close"].values   # NumPy 数组形状形如： (2264, 1)

  train_dataset, val_dataset = prepare_datasets(
      series=time_series,
      context_length=context_len,
      horizon_length=horizon_len,
      freq_type=freq_type,
      train_split=0.8,
  )

  print(f"Created datasets:")
  print(f"- Training samples: {len(train_dataset)}")
  print(f"- Validation samples: {len(val_dataset)}")
  print(f"- Using frequency type: {freq_type}")
  return train_dataset, val_dataset

def get_news_dataset_data(context_len: int, horizon_len: int, freq_type: int = 0):
    """
    Load time series data and corresponding text data from CSV files.

    Args:
        context_len: Number of past timesteps to use
        horizon_len: Number of future timesteps to predict
        freq_type: Frequency type (0, 1, or 2)

    Returns:
        Tuple of (train_dataset, val_dataset, train_texts, val_texts)
    """
    # 读取时间序列数据
    ts_file = "/home/zzx/projects/rrg-timsbc/zzx/timesfm/News dataset/Facebook_Obama_transpose_final.csv"
    ts_df = pd.read_csv(ts_file, index_col=0)  # IDLink 作为索引
    ts_df = ts_df.T  # 转置，使行表示新闻，列表示时间步
    ts_data = ts_df.values  # 转换为 NumPy 数组

    # 读取文本数据
    txt_file = "/home/zzx/projects/rrg-timsbc/zzx/timesfm/News dataset/Facebook_Obama_txt_final.csv"
    txt_df = pd.read_csv(txt_file, sep=",")
    
    # **确保 ID 类型一致**
    txt_df["id"] = txt_df["id"].astype(str)  # 确保 ID 是字符串
    ts_df.columns = ts_df.columns.astype(str)  # 确保时间序列数据的 ID 也是字符串
    ts_df.columns = [col.split('.')[0] for col in ts_df.columns]

    # **仅保留时间序列文件中存在的 ID**
    txt_df = txt_df[txt_df["id"].isin(ts_df.columns)]
    
    # 确保顺序一致
    txt_df = txt_df.set_index("id").loc[ts_df.columns]
    texts = txt_df["text"].tolist()  # 提取文本数据

    # 构建训练和验证集
    train_dataset, val_dataset, train_texts, val_texts = prepare_datasets(
        series=ts_data,
        texts=texts,
        context_length=context_len,
        horizon_length=horizon_len,
        freq_type=freq_type
    )

    print(f"Created datasets:")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Validation samples: {len(val_dataset)}")
    print(f"- Using frequency type: {freq_type}")

    return train_dataset, val_dataset, train_texts, val_texts


def single_gpu_example():
  """Basic example of finetuning TimesFM on stock data."""
  timesfm_model, mm_timesfm_model, hparams, tfm_config = get_models(load_weights=True)
  config = FinetuningConfig(batch_size=256,
                            num_epochs=5,
                            learning_rate=1e-4,
                            use_wandb=True,
                            freq_type=1,
                            log_every_n_steps=10,
                            val_check_interval=0.5,
                            use_quantile_loss=True)

  # train_dataset, val_dataset = get_data(128,
  #                                       tfm_config.horizon_len,
  #                                       freq_type=config.freq_type)
  
  train_dataset, val_dataset, train_texts, val_texts = get_news_dataset_data(128,
                                                                tfm_config.horizon_len,
                                                                freq_type=config.freq_type)
  
  finetuner = TimesFMFinetuner(model=timesfm_model, config=config,MMTimesFM_model=mm_timesfm_model)

  print("\nStarting finetuning MMTimesFM...")
  results = finetuner.finetune(train_dataset=train_dataset,
                               val_dataset=val_dataset,
                               train_texts=train_texts,
                               val_texts=val_texts)

  print("\nFinetuning completed!")
  print(f"Training history: {len(results['history']['train_loss'])} epochs")

  plot_predictions(
      model=mm_timesfm_model,
      val_dataset=val_dataset,
      save_path="mm_timesfm_predictions.png",
  )


def setup_process(rank, world_size, mm_timesfm_model, timesfm_model, config, train_dataset, val_dataset, train_texts, val_texts,
                  return_dict):
  """Setup process function with optimized CUDA handling."""
  try:
    if torch.cuda.is_available():
      torch.cuda.set_device(rank)

    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = config.master_port
    if not torch.distributed.is_initialized():
      torch.distributed.init_process_group(backend="nccl",
                                           world_size=world_size,
                                           rank=rank)

    finetuner = TimesFMFinetuner(mm_timesfm_model, config, rank=rank)

    results = finetuner.finetune(train_dataset=train_dataset,
                                 val_dataset=val_dataset,
                                 train_texts=train_texts,
                                 val_texts=val_texts)

    if rank == 0:
      return_dict["results"] = results
      plot_predictions(
          model=mm_timesfm_model,
          val_dataset=val_dataset,
          save_path="timesfm_predictions.png",
      )

  except Exception as e:
    print(f"Error in process {rank}: {str(e)}")
    raise e
  finally:
    if torch.distributed.is_initialized():
      torch.distributed.destroy_process_group()


def multi_gpu_example():
  """Example of finetuning TimesFM using multiple GPUs with optimized spawn."""
  mp.set_start_method("spawn", force=True)

  gpu_ids = [0, 1]
  world_size = len(gpu_ids)

  timesfm_model, mm_timesfm_model, hparams, tfm_config = get_models(load_timesfm_weights=True)

  # Create config
  config = FinetuningConfig(
      batch_size=256,
      num_epochs=5,
      learning_rate=3e-5,
      use_wandb=True,
      distributed=True,
      gpu_ids=gpu_ids,
      log_every_n_steps=50,
      val_check_interval=0.5,
  )
  # train_dataset, val_dataset = get_data(128, tfm_config.horizon_len)
  train_dataset, val_dataset, train_texts, val_texts = get_news_dataset_data(128,
                                                                tfm_config.horizon_len,
                                                                freq_type=config.freq_type)
  
  manager = mp.Manager()
  return_dict = manager.dict()

  # Launch processes
  mp.spawn(
      setup_process,
      args=(world_size, mm_timesfm_model, timesfm_model, config, train_dataset, val_dataset, train_texts, val_texts, return_dict),
      nprocs=world_size,
      join=True,
  )

  results = return_dict.get("results", None)
  print("\nFinetuning completed!")
  return results


def main(argv):
  """Main function that selects and runs the appropriate training mode."""

  try:
    if FLAGS.training_mode == "single":
      print("\nStarting single-GPU training...")
      single_gpu_example()
    else:
      gpu_ids = [int(id) for id in FLAGS.gpu_ids]
      print(f"\nStarting multi-GPU training using GPUs: {gpu_ids}...")

      config = FinetuningConfig(
          batch_size=256,
          num_epochs=5,
          learning_rate=3e-5,
          use_wandb=True,
          distributed=True,
          gpu_ids=gpu_ids,
      )

      results = multi_gpu_example(config)
      print("\nMulti-GPU training completed!")

  except Exception as e:
    print(f"Training failed: {str(e)}")
  finally:
    if torch.distributed.is_initialized():
      torch.distributed.destroy_process_group()


if __name__ == "__main__":
  app.run(main)

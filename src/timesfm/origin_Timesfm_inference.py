import sys
from threading import local
print(sys.executable)
print("Python version3.11")
import timesfm
import pandas as pd

# # Loading the timesfm-2.0 checkpoint:
# # For PAX
# tfm = timesfm.TimesFm(
#       hparams=timesfm.TimesFmHparams(
#           backend="gpu",
#           per_core_batch_size=32,
#           horizon_len=128,
#           num_layers=50,
#           context_len=2048,

#           use_positional_embedding=False,
#       ),
#       checkpoint=timesfm.TimesFmCheckpoint(
#           huggingface_repo_id="google/timesfm-2.0-500m-jax"),
#   )

# checkpoint=timesfm.TimesFmCheckpoint(
#         huggingface_repo_id="google/timesfm-2.0-500m-pytorch",
#         local_dir="/home/zzx/projects/rrg-timsbc/zzx/torch_model.ckpt")
# print(checkpoint.path)

# For Torch
tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="gpu",
          per_core_batch_size=32,
          horizon_len=128,
          num_layers=50,
          use_positional_embedding=False,
          context_len=2048,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
        # huggingface_repo_id="google/timesfm-2.0-500m-pytorch",
        path="/home/zzx/projects/rrg-timsbc/zzx/torch_model.ckpt"),
  )

print("TimesFm loaded successfully!")

# # Inference
csv_file_path = '/home/zzx/projects/rrg-timsbc/zzx/News dataset/Facebook_Obama_transpose_final.csv'
input_df = pd.read_csv(csv_file_path)

forecast_df = tfm.forecast_on_df(
    inputs=input_df,
    freq="MIN",  # minutely
    value_name="61874",
    num_jobs=-1,
)
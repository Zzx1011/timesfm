from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from timesfm import timesfm_torch
from timesfm import pytorch_patched_decoder as ppd  # 引入 PatchedTimeSeriesDecoder
from timesfm.pytorch_patched_decoder import TimesFMDecoderLayer, causal_mask, merge_masks, convert_paddings_to_mask
from typing import List, Tuple

class StackedDecoder(nn.Module):
  """Stacked transformer layer."""

  def __init__(
      self,
      hidden_size: int,
      intermediate_size: int,
      num_heads: int,
      num_kv_heads: int,
      head_dim: int,
      num_layers: int,
      rms_norm_eps: float = 1e-6,
  ):
    super().__init__()

    self.layers = nn.ModuleList()
    for _ in range(num_layers):
      self.layers.append(
          TimesFMDecoderLayer(
              hidden_size=hidden_size,
              intermediate_size=intermediate_size,
              num_heads=num_heads,
              num_kv_heads=num_kv_heads,
              head_dim=head_dim,
              rms_norm_eps=rms_norm_eps,
          ))

  def forward(
      self,
      hidden_states: torch.Tensor,
      paddings: torch.Tensor,
      kv_write_indices: torch.Tensor | None = None,
      kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] | None = None,
  ) -> torch.Tensor:
    padding_mask = convert_paddings_to_mask(paddings, hidden_states.dtype)
    atten_mask = causal_mask(hidden_states)
    
    if torch.isnan(padding_mask).any() or torch.isinf(padding_mask).any():
        print("Warning: NaN or Inf detected in padding_mask.")   
    if torch.isnan(atten_mask).any() or torch.isinf(atten_mask).any():
        print("Warning: NaN or Inf detected in atten_mask.")
    mask = merge_masks(padding_mask, atten_mask)
    for i in range(len(self.layers)):
      layer = self.layers[i]
      kv_cache = kv_caches[i] if kv_caches is not None else None
      _, hidden_states = layer(
          hidden_states=hidden_states,
          mask=mask,
          paddings=paddings,
          kv_write_indices=kv_write_indices,
          kv_cache=kv_cache,
      )
    return hidden_states
  
class MMTimesFM(ppd.PatchedTimeSeriesDecoder):
    def __init__(self, text_model_name="bert-base-uncased", **kwargs):
        super().__init__(**kwargs)
        
        self.config = kwargs['config']
        # 1. 加载预训练的文本模型 (BERT)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        
        # 2. 线性变换到 TimesFM 需要的维度
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, self.config.hidden_size)
        
        # 3. Cross-Attention 融合 text_embedding 和 time_series_embedding
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.config.hidden_size, num_heads=8, batch_first=True)

        # 4. 复用 TimesFM 的核心部分
        self.decoder = ppd.PatchedTimeSeriesDecoder(self.config)

        self.stacked_transformer = StackedDecoder(
        hidden_size=self.config.hidden_size,
        intermediate_size=self.config.intermediate_size,
        num_heads=self.config.num_heads,
        num_kv_heads=self.config.num_kv_heads,
        head_dim=self.config.head_dim,
        num_layers=self.config.num_layers,
        rms_norm_eps=self.config.rms_norm_eps,
    )

    def encode_text(self, text_batch):
        """将文本转换为 text_embedding, 支持 batch 处理"""
        if text_batch is None:
            return None
        tokens = self.tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_embedding = self.text_encoder(**tokens).last_hidden_state  # (batch, text_len, hidden_size)
        return self.text_proj(text_embedding)  # (batch, text_len, model_dims)

    def forward(self, input_ts, input_padding, freq, text=None):
        """
        input_ts: (batch, context_len, model_dims) 时间序列输入
        input_padding: (batch, context_len) Padding Mask
        freq: (batch, 1) 频率编码
        text: (str) 输入文本
        """
        # 1. 计算 text_embedding
        text_embedding = self.encode_text(text)  # (batch, text_len, model_dims)
        
        # 2. 计算 time_series_embedding
        model_input, patched_padding, stats, _ = self.decoder._preprocess_input(input_ts, input_padding)
        print("patched_padding.shape: ",patched_padding.shape)
        print("patched_padding: ",patched_padding)

        # 3. 用 Cross-Attention 融合文本特征
        if text_embedding is not None:
            fused_embedding, _ = self.cross_attention(query=model_input, key=text_embedding, value=text_embedding)
        else:
            fused_embedding = model_input

        f_emb = self.freq_emb(freq)  # B x 1 x D
        fused_embedding = fused_embedding + f_emb

        # 4. 传入 TimesFM 的核心部分 (PatchedTimeSeriesDecoder)
        model_output = self.decoder.stacked_transformer(fused_embedding, patched_padding)
        print("model_output.shape",model_output.shape)
        if torch.isnan(model_output).any() or torch.isinf(model_output).any():
            print("Warning: NaN or Inf detected in model_output.")

        # 5. 还原输出
        return self.decoder._postprocess_output(model_output, len(self.config.quantiles) + 1, stats)
    
    def decode(self, input_ts, paddings, freq, horizon_len, output_patch_len=None, max_len=None, return_forecast_on_context=False, text=None):
        """支持文本输入的自回归解码过程"""
        
        if text is None:
            return super().decode(input_ts, paddings, freq, horizon_len, output_patch_len, max_len, return_forecast_on_context)

        final_out = input_ts  # (batch, context_len, model_dims)
        context_len = final_out.shape[1]
        full_outputs = []

        if max_len is None:
            max_len = context_len

        if paddings.shape[1] != final_out.shape[1] + horizon_len:
            raise ValueError(
                "Length of paddings must match length of input + horizon_len:"
                f" {paddings.shape[1]} != {final_out.shape[1]} + {horizon_len}")

        if output_patch_len is None:
            output_patch_len = self.decoder.config.horizon_len

        num_decode_patches = (horizon_len + output_patch_len - 1) // output_patch_len

        for step_index in range(num_decode_patches):
            current_padding = paddings[:, 0:final_out.shape[1]]
            input_ts = final_out[:, -max_len:]
            input_padding = current_padding[:, -max_len:]

            fprop_outputs = self.forward(input_ts, input_padding)

            if return_forecast_on_context and step_index == 0:
                fprop_outputs = self.forward(input_ts, input_padding, text=text) # 第一步解码用文本信息进行预测
                new_full_ts = fprop_outputs[:, 0:-1, 0:self.decoder.config.patch_len, :]
                new_full_ts = new_full_ts.reshape(new_full_ts.size(0), -1, new_full_ts.size(3))
                full_outputs.append(new_full_ts)

            new_ts = fprop_outputs[:, -1, :output_patch_len, 0]
            new_full_ts = fprop_outputs[:, -1, :output_patch_len, :]

            full_outputs.append(new_full_ts)
            final_out = torch.cat([final_out, new_ts], axis=1)

        if return_forecast_on_context:
            full_outputs = torch.cat(full_outputs, axis=1)[:, :(context_len - self.decoder.config.patch_len + horizon_len), :]
        else:
            full_outputs = torch.cat(full_outputs, axis=1)[:, 0:horizon_len, :]

        return (full_outputs[:, :, 0], full_outputs)
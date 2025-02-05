from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from timesfm import timesfm_torch
from timesfm import pytorch_patched_decoder as ppd  # 引入 PatchedTimeSeriesDecoder

class MMTimesFm(ppd.PatchedTimeSeriesDecoder):
    def __init__(self, text_model_name="bert-base-uncased", **kwargs):
        super().__init__(**kwargs)
        
        # 1. 加载预训练的文本模型 (BERT)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        
        # 2. 线性变换到 TimesFM 需要的维度
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, self.model_dims)
        
        # 3. Cross-Attention 融合 text_embedding 和 time_series_embedding
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.model_dims, num_heads=8, batch_first=True)

        # 4. 复用 TimesFM 的核心部分
        self.decoder = ppd.PatchedTimeSeriesDecoder(self._model_config)

    def encode_text(self, text):
        """将文本转换为 text_embedding"""
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_embedding = self.text_encoder(**tokens).last_hidden_state  # (batch, text_len, hidden_size)
        return self.text_proj(text_embedding)  # 变换到 model_dims

    def forward(self, input_ts, input_padding, freq, text):
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

        # 3. 用 Cross-Attention 融合文本特征
        fused_embedding, _ = self.cross_attention(query=model_input, key=text_embedding, value=text_embedding)

        # 4. 传入 TimesFM 的核心部分 (PatchedTimeSeriesDecoder)
        model_output = self.decoder.stacked_transformer(fused_embedding, patched_padding)

        # 5. 还原输出
        return self.decoder._postprocess_output(model_output, len(self._model_config.quantiles) + 1, stats)

    def decode(self, input_ts, paddings, freq, horizon_len, output_patch_len=None, max_len=None, return_forecast_on_context=False, text=None):
        """支持文本输入的解码过程"""
        if text is None:
            return super().decode(input_ts, paddings, freq, horizon_len, output_patch_len, max_len, return_forecast_on_context)
        return self.forward(input_ts, paddings, freq, text)
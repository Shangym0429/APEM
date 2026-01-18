from transformers import BertModel
import torch.nn as nn

class TextEncoder(nn.Module):

    def __init__(self, bert_model_name='agriculture-bert-uncased', max_text_tokens=max_text_tokens):
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_size = self.bert.config.hidden_size  # 768
        self.max_text_tokens = max_text_tokens
        self.text_pooler = nn.AdaptiveAvgPool1d(max_text_tokens) if max_text_tokens > 0 else None


    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        text_global = outputs.last_hidden_state[:, 0, :]  # [B, 768]

        text_seq = outputs.last_hidden_state  # [B, seq_len, 768]

        if self.text_pooler is not None:
            text_seq = self.text_pooler(text_seq.permute(0, 2, 1)).permute(0, 2, 1)  # [B, max_text_tokens, 768]

        return text_global, text_seq
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
from transformers import BertConfig
import torch.nn.functional as F
from Model_CharBERT import CharBERTModel
from Multiple_attention import TAMM


class CharBertModel(nn.Module):
    def __init__(self, device):
        super(CharBertModel, self).__init__()
        config = BertConfig.from_pretrained('charbert-bert-wiki')
        self.bert = CharBERTModel(config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(768, 2)
        self.hidden_size = 768
        self.fuse = nn.Conv1d(2 * self.hidden_size, self.hidden_size, kernel_size=1)
        self.model_tamm = TAMM(channel=12).to(device)
        self.device = device

    def forward(self, x):
        context = x[0]
        types = x[1]
        mask = x[2]
        char_ids = x[3]
        start_ids = x[4]
        end_ids = x[5]

        all_hidden_states_word, all_hidden_states_char, pooled_output = self.bert(
            char_input_ids=char_ids,
            start_ids=start_ids, end_ids=end_ids,
            input_ids=context,
            attention_mask=mask,
            token_type_ids=types,
            output_hidden_states=True
        )

        fuse_output = []
        for x1, x2 in zip(all_hidden_states_word, all_hidden_states_char):
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            x = torch.cat([x1, x2], dim=-1)
            x = x.view(x.size(0), -1, x.size(2))
            y = self.fuse(x.transpose(1,2))
            y_output = y.transpose(1, 2)
            fuse_output.append(y_output)

        pyramid = tuple(fuse_output)
        pyramid = torch.stack(pyramid, dim=0).permute(1, 0, 2, 3)
        pos_pooled = self.model_tamm.forward(pyramid)
        compressed_feature_tensor = torch.mean(pos_pooled, dim=2)
        compressed_feature_tensor = torch.mean(compressed_feature_tensor, dim=1)
        out = self.dropout(compressed_feature_tensor)
        out = self.fc(out)
        return pyramid, pooled_output, out

class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained("charbert-bert-wiki")
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(768, 2)
        self.model_tamm = TAMM(channel=12).to(device)
        self.device = device

    def forward(self, x):
        context = x[0]
        types = x[1]
        mask = x[2]
        outputs, pooled = self.bert(input_ids=context, token_type_ids=types,
                                    attention_mask=mask,
                                    output_all_encoded_layers=True)
        pyramid = tuple(outputs)
        pyramid = torch.stack(pyramid, dim=0).permute(1, 0, 2, 3)
        pos_pooled = self.model_tamm.forward(pyramid)
        compressed_feature_tensor = torch.mean(pos_pooled, dim=2)
        compressed_feature_tensor = torch.mean(compressed_feature_tensor, dim=1)
        out = self.dropout(compressed_feature_tensor)
        out = self.fc(out)
        return pyramid, pooled, out
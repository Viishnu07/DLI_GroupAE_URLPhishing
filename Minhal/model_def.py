%%writefile model_def.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel

# ============= MULTI-LAYER FUSION =============
class FastMultiLayerFusion(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.tensor([0.2, 0.3, 0.5]))
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, all_hidden_states):
        if len(all_hidden_states) == 7:  # DistilBERT: 6 layers + embedding
            key_layers = [all_hidden_states[2], all_hidden_states[4], all_hidden_states[6]]
        else:
            n_layers = len(all_hidden_states)
            indices = [min(2, n_layers-1), n_layers // 2, n_layers-1]
            key_layers = [all_hidden_states[i] for i in indices]

        weights = F.softmax(self.layer_weights, dim=0)
        fused = sum(w * layer[:, 0, :] for w, layer in zip(weights, key_layers))
        return self.dropout(self.transform(fused))

# ============= MULTI-SCALE MODULE =============
class FastMultiScale(nn.Module):
    def __init__(self, in_channels=768, out_channels=256):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels//2, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels, out_channels//2, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.transpose(1, 2)
        f1 = self.conv1(x)
        f3 = self.conv3(x)
        multi_scale = torch.cat([f1, f3], dim=1)
        multi_scale = self.batch_norm(multi_scale)
        max_pool = F.max_pool1d(multi_scale, kernel_size=multi_scale.size(2)).squeeze(2)
        avg_pool = F.avg_pool1d(multi_scale, kernel_size=multi_scale.size(2)).squeeze(2)
        return self.dropout(torch.cat([max_pool, avg_pool], dim=1))

# ============= MAIN MODEL =============
class FastTransURLModel(nn.Module):
    def __init__(self, num_classes=2, model_name='distilbert-base-uncased'):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name, output_hidden_states=True)

        for param in self.distilbert.parameters():
            param.requires_grad = False
        for param in self.distilbert.transformer.layer[-3:].parameters():
            param.requires_grad = True

        self.layer_fusion = FastMultiLayerFusion(hidden_size=768)
        self.multi_scale = FastMultiScale(in_channels=768, out_channels=256)

        self.classifier = nn.Sequential(
            nn.Linear(768 + 512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        self.use_auxiliary = True
        if self.use_auxiliary:
            self.aux_classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states
        fused_features = self.layer_fusion(all_hidden_states)
        last_hidden = outputs.last_hidden_state
        multi_scale_features = self.multi_scale(last_hidden)
        combined = torch.cat([fused_features, multi_scale_features], dim=1)
        logits = self.classifier(combined)

        if self.use_auxiliary and self.training:
            aux_logits = self.aux_classifier(fused_features)
            return logits, aux_logits

        return logits

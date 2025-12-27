from src.requirements import *

class SpecAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mask = T.TimeMasking(time_mask_param=8)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=4)

    def forward(self, x):
        # only works in train() mode
        if not self.training:
            return x

        x = x.transpose(1, 2)
        x = self.time_mask(x)
        x = self.freq_mask(x)
        x = x.transpose(1, 2)

        return x        

class ASRModel(nn.Module):
    def __init__(self, ssl_model, vocab_size, freeze_ssl):
        super().__init__()
        self.model = ssl_model

        if freeze_ssl:
            for module in [self.model.encoder, self.model.context]:
                for p in module.parameters():
                    p.requires_grad = False
        else:
            for module in [self.model.encoder, self.model.context]:
                for p in module.parameters():
                    p.requires_grad = True

        for module in [self.model.encoder_m, self.model.context_m, self.model.target_proj_m]:
            for p in module.parameters():
                p.requires_grad = False

        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()
                module.track_running_stats = False

        self.spec_augment = SpecAugment()
        
        self.norm = nn.LayerNorm(128)
        
        # self.decoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         d_model = 128,
        #         nhead = 4,
        #         dim_feedforward = 512,
        #         dropout = 0.1,
        #         batch_first = True
        #     ),
        #     num_layers = 6
        # )

        self.decoder = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        self.proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Dropout(0.1)
        )
        self.fc = nn.Linear(256, vocab_size + 1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def create_padding_mask(self, features, lengths):
        batch_size, max_len, _ = features.size()

        indices = torch.arange(max_len).to(features.device)
        mask = indices.unsqueeze(0) >= lengths.unsqueeze(1)

        return mask

    def forward(self, x, lengths=None):
        c = self.model.extract_features(x)

        padding_mask = None
        if lengths is not None:
            padding_mask = self.create_padding_mask(c, lengths)
        
        c = self.spec_augment(c)
        c = self.norm(c)
        # c = self.decoder(c, src_key_padding_mask=padding_mask)
        c, _ = self.decoder(c)
        c = self.proj(c)
        
        logits = self.fc(c)
        log_probs = self.log_softmax(logits)
        
        return log_probs

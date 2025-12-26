from src.requirements import *

class SpecAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mask = T.TimeMasking(time_mask_params=20)
        self.freq_mask = T,FrequencyMasking(freq_mask_param=16)

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
        
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model = 128,
                nhead = 4,
                dim_feedforward = 512,
                dropout = 0.1,
                batch_first = True
            ),
            num_layers = 6
        )

        self.proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.fc = nn.Linear(512, vocab_size + 1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        c = self.model.extract_features(x)
        c = self.spec_augment(c)

        c = self.norm(c)
        c = self.decoder(c)

        logits = self.fc(c)
        log_probs = self.log_softmax(logits)
        
        return log_probs

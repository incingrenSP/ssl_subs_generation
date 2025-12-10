from src.requirements import *

class ASRModel(nn.Module):
    def __init__(self, ssl_model, vocab_size, freeze_ssl=True):
        super().__init__()
        self.model = ssl_model

        if frozen_ssl:
            for module in [self.model.encoder, self.model.context]:
                for p in module.parameters():
                    p.requires_grad = False
        else:
            for module in [self.model.encoder, self.model.context]:
                for p in module.parameters():
                    p.requires_grad = True

        for module in [self.model.encoder_m, self.model.context_m, self.model.target_proj_m]:
            for p in module.parameters():
                module.requires_grad = False

        self.norm = nn.LayerNorm(128)
        
        self.decoder_rnn = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(512, vocab_size + 1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def extract_features(self, x):
        z = self.encoder(x)
        z = z.transpose(1, 2)
        return self.context(z)

    def forward(self, x):
        c = self.model.extract_features(x)

        c = self.norm(c)
        c, _ = self.decoder_rnn(c)

        logits = self.fc(c)
        log_probs = self.log_softmax(logits)
        
        return log_probs

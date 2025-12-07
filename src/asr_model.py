from src.requirements import *

class ASRModel(nn.Module):
    def __init__(self, ssl_model, vocab_size, freeze_ssl=True):
        super().__init__()
        self.model = ssl_model

        if freeze_ssl:
            for p in self.model.encoder.parameters():
                p.requires_grad = False
        else:
            for p in self.model.encoder.parameters():
                p.requires_grad = True

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

    def forward(self, x):
        z = self.model.encoder(x)
        z = z.transpose(1, 2)
        c = self.model.context(z)

        c = self.norm(c)
        c, _ = self.decoder_rnn(c)

        logits = self.fc(c)
        log_probs = self.log_softmax(logits)
        
        return logits

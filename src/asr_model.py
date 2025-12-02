from src.requirements import *

class ASRModel(nn.Module):
    def __init__(self, ssl_model, vocab_size, freeze_ssl=True):
        super().__init__()
        self.model = ssl_model
        if freeze_ssl:
             for params in self.model.encoder.parameters():
                 params.requires_grad = False

        self.fc = nn.Linear(128, vocab_size+1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        z = self.model.encoder(x)
        z = z.transpose(1, 2)

        c = self.model.context(z)

        logits = self.fc(c)
        log_probs = self.log_softmax(logits)

        return log_probs
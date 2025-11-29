from src.requirements import *

class Tokenizer:
    def __init__(self, corpus_path, add_blank=True, vocab=None):
        if vocab is not None:
            self.vocab = vocab
            self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
            self.id_to_token = {i: t for t, i in self.token_to_id.items()}
            return
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = [unicodedata.normalize('NFKC', l.strip()) for l in f]

        tokens = []
        for line in lines:
            tokens.extend(self.tokenize(line))

        counter = Counter(tokens)
        self.vocab = sorted(counter.keys())

        if add_blank:
            self.vocab = ['<blank>'] + self.vocab

        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def tokenize(self, text):
        text = unicodedata.normalize('NFKC', text)
        CLEANUP_PATTERN = re.compile(r'[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u206f\ufeff\u00ad\u0000-\u001f]')
        cleaned_text = CLEANUP_PATTERN.sub('', text)
        return regex.findall(r'\X', cleaned_text)

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.token_to_id[t] for t in tokens if t in self.token_to_id]

    def decode(self, ids):
        return ''.join([self.id_to_token[i] for i in ids if i in self.id_to_token])

    def save(self, path):
        data = {"vocab": self.vocab}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Tokenizer(corpus_path=None, vocab=data["vocab"])

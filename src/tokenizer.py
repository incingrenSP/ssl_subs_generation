from src.requirements import *

class Tokenizer:
    def __init__(self, vocab=None):
        self.blank_token = "<blank>"
        self.blank_id = 0
        
        if vocab is not None:
            self.token_to_id = vocab["tokens"]
            self.id_to_token = {int(k): v for k, v in vocab["ids"].items()}
            self.vocab_size = vocab["size"]

            return

        self.token_to_id = {}
        self.id_to_token = {}

    # Normalization
    def normalize(self, text: str) -> str:
        return unicodedata.normalize("NFD", text)

    def denormalize(self, text: str) -> str:
        return unicodedata.normalize("NFC", text)

    # Vocab
    def build_vocab(self, texts):
        counter = Counter()

        for text in texts:
            text = self.normalize(text)
            for ch in text:
                counter[ch] += 1

        # these seem to be missing in corpus so just hardcoding these
        required_chars = (
            "०१२३४५६७८९" 
            ".,?!:;()\"'- " +
            "।॥"
        )
        
        # <blank> : id[0]
        self.token_to_id = {self.blank_token: self.blank_id}
        self.id_to_token = {self.blank_id: self.blank_token}

        next_id = 1
        for ch in required_chars:
            if ch not in self.token_to_id:
                self.token_to_id[ch] = next_id
                self.id_to_token[next_id] = ch
                next_id += 1
                
        for ch, count in counter.most_common():
            if ch not in self.token_to_id:
                self.token_to_id[ch] = next_id
                self.id_to_token[next_id] = ch
                next_id += 1

        self.vocab_size = next_id

    # Encoding / Decoding
    def encode(self, text: str):
        text = self.normalize(text)
        ids = []

        for ch in text:
            if ch not in self.token_to_id:
                raise ValueError(f"Unknown character: {repr(ch)}")
            ids.append(self.token_to_id[ch])

        return ids

    def decode(self, ids):
        chars = []

        for i in ids:
            if i == self.blank_id:
                continue
            chars.append(self.id_to_token[i])

        text = "".join(chars)
        return self.denormalize(text)

    def save(self, path):
        data = {"tokens" : self.token_to_id, "ids" : self.id_to_token, "size" : self.vocab_size}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Tokenizer(vocab=data)

    def get_vocab(self):
        vocab = []
        for k, v in self.token_to_id.items():
            vocab.append(k)

        return vocab

    def __len__(self):
        return self.vocab_size
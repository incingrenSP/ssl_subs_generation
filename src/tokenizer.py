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
            
        CHARACTER_WHITELIST = set([
            'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ',
            'ए', 'ऐ', 'ओ', 'औ', 'अं', 'अः',
            'क', 'ख', 'ग', 'घ', 'ङ',
            'च', 'छ', 'ज', 'झ', 'ञ',
            'ट', 'ठ', 'ड', 'ढ', 'ण',
            'त', 'थ', 'द', 'ध', 'न',
            'प', 'फ', 'ब', 'भ', 'म',
            'य', 'र', 'ल', 'व',
            'श', 'ष', 'स', 'ह', 'क्ष', 'त्र', 'ज्ञ',
            'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'े', 'ैे', 'ो', 'ौ', 'ँ', 'ं', 'ः',
            '०', '१', '२', '३', '४', '५', '६', '७', '८', '९',
            '।', ',', '.', '?', '!', '"', "'", '-',
            ' '
        ])

        filtered_tokens = [token for token in tokens if token in CHARACTER_WHITELIST]
        counter = Counter(filtered_tokens)
        self.vocab = sorted(counter.keys())

        if add_blank:
            self.vocab = ['<blank>'] + self.vocab

        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

        print(f"Final Vocabulary Size after filtering: {len(self.vocab)}")
        print(f"Blank ID: {self.token_to_id.get('<blank>', 'Not Found')}")

    def tokenize(self, text):
        text = unicodedata.normalize('NFKC', text)
        CLEANUP_PATTERN = re.compile(r'[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u206f\ufeff\u00ad\u0000-\u001f]')
        cleaned_text = CLEANUP_PATTERN.sub('', text)

        parts = cleaned_text.split(' ')
        tokens = []
        space_token = ' '

        for i, part in enumerate(parts):
            if part:
                graphemes = regex.findall(r'\X', part)
                tokens.extend(graphemes)

            if i < len(parts) - 1:
                tokens.append(space_token)
        
        return tokens

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
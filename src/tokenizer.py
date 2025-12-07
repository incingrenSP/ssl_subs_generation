from src.requirements import *

class Tokenizer:
    def __init__(self, corpus_path, add_blank=True, vocab=None):
        if vocab is not None:
            self.vocab = vocab
            self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
            self.id_to_token = {i: t for t, i in self.token_to_id.items()}
            print(f"Final Vocabulary Size after filtering: {len(self.vocab)}")
            print(f"Blank ID: {self.token_to_id.get('<blank>', 'Not Found')}")
            return
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = [unicodedata.normalize('NFKC', l.strip()) for l in f]

        tokens = []
        for line in lines:
            tokens.extend(self.tokenize(line))

        vowels = [
            'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 
            'ए', 'ऐ', 'ओ', 'औ', 'अं', 'अः',
        ]
        special_chars = [
            '०', '१', '२', '३', '४', '५', '६', '७', '८', '९', '।', ',', '?', '!', '-', '(', ')', '"', "'", ' '
        ]
        consonants = [
            'क', 'ख', 'ग', 'घ', 'ङ',
            'च', 'छ', 'ज', 'झ', 'ञ',
            'ट', 'ठ', 'ड', 'ढ', 'ण',
            'त', 'थ', 'द', 'ध', 'न',
            'प', 'फ', 'ब', 'भ', 'म',
            'य', 'र', 'ल', 'व',
            'श', 'ष', 'स', 'ह',
            'क्ष', 'त्र', 'ज्ञ'
        ]
        matras = [
            "", "ा", "ि", "ी", "ु", "ू", "े",
            "ै", "ो", "ौ", "ं", "ः", "ँ", "्"
        ]
        # half_letters = []

        CHARACTER_WHITELIST = set(vowels + self._gen_vocab(consonants, matras) + special_chars)
        
        filtered_tokens = [token for token in tokens if token in CHARACTER_WHITELIST]
        counter = Counter(filtered_tokens)
        self.vocab = sorted(counter.keys())

        if add_blank:
            self.vocab = ['<blank>'] + self.vocab

        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

        print(f"Final Vocabulary Size after filtering: {len(self.vocab)}")
        print(f"Blank ID: {self.token_to_id.get('<blank>', 'Not Found')}")

    def _gen_vocab(self, consonants, matras):
        vocabs = []
        for c in consonants:
            for m in matras:
                vocabs.append(c + m)
        return vocabs

    def _split_grapheme(self, token):
        sp_keyword = '्'
        if sp_keyword not in token:
            return [token]
        parts = []
        temp = ''
    
        for char in token:
            temp += char
            if char in sp_keyword:
                parts.append(temp)
                temp = ''
    
        if temp:
            parts.append(temp)
    
        return parts

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
                for g in graphemes:
                    tokens.extend(self._split_grapheme(g))
        
            if i < len(parts) - 1:
                tokens.append(space_token)
                
        return tokens

    def encode(self, text):
        tokens = self.tokenize(text)
        output = [self.token_to_id[t] for t in tokens if t in self.token_to_id]
        return output

    def decode(self, ids):
        output = ''.join([self.id_to_token[i] for i in ids])
        return output

    def save(self, path):
        data = {"vocab": self.vocab}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Tokenizer(corpus_path=None, vocab=data["vocab"])
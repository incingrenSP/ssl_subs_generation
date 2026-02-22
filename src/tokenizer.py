import json
import re
from pathlib import Path
from typing import List, Optional
import unicodedata

from src.requirements import *

class NepaliTokenizer:
    # Junk Unicode characters to remove
    JUNK_CHARS = {
        '\u200d',  # Zero-width joiner
        '\u200c',  # Zero-width non-joiner
        '\u200b',  # Zero-width space
        '\ufeff',  # Zero-width no-break space (BOM)
        '\u200e',  # Left-to-right mark
        '\u200f',  # Right-to-left mark
        '\xa0',    # Non-breaking space
    }
    
    # Special tokens
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    BLANK_TOKEN = '<blank>'  # CTC blank
    SPACE_TOKEN = '<space>'
    
    def __init__(self, vocab_size: Optional[int] = None, add_space_token: bool = True):
        self.vocab_size = vocab_size
        self.add_space_token = add_space_token
        
        self.char2id = {}
        self.id2char = {}
        
        # Special token IDs (CTC blank MUST be 0)
        self.blank_id = 0
        self.pad_id = 1
        self.unk_id = 2
        
        self._add_special_tokens()
        self._vocab_built = False
    
    def _add_special_tokens(self):
        special_tokens = [
            self.BLANK_TOKEN,  # ID: 0 (CTC blank)
            self.PAD_TOKEN,    # ID: 1
            self.UNK_TOKEN,    # ID: 2
        ]
        
        if self.add_space_token:
            special_tokens.append(self.SPACE_TOKEN)  # ID: 3
        
        for token in special_tokens:
            self._add_token(token)
    
    def _add_token(self, token: str):
        if token not in self.char2id:
            token_id = len(self.char2id)
            self.char2id[token] = token_id
            self.id2char[token_id] = token
    
    def clean_text(self, text: str) -> str:
        # Remove junk characters
        for junk_char in self.JUNK_CHARS:
            text = text.replace(junk_char, '')
        
        # Normalize Unicode (NFC normalization for Devanagari)
        text = unicodedata.normalize('NFC', text)
        
        # Normalize whitespace (collapse multiple spaces, strip)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _is_devanagari(self, char: str) -> bool:
        """Check if character is in Devanagari Unicode block"""
        if len(char) != 1:
            return False
        
        code_point = ord(char)
        # Devanagari block: U+0900 to U+097F
        # Devanagari Extended: U+A8E0 to U+A8FF
        return (0x0900 <= code_point <= 0x097F) or (0xA8E0 <= code_point <= 0xA8FF)
    
    def build_vocab_from_metadata(self, metadata_path: str, min_freq: int = 1):
        # Load metadata
        metadata = pd.read_csv(
            metadata_path,
            sep='\t',
            names=['path', 'transcript'],
            header=0 if self._has_header(metadata_path) else None,
        )
        
        # Count character frequencies
        char_freq = Counter()
        
        for transcript in metadata['transcript']:
            cleaned_text = self.clean_text(str(transcript))
            
            # Handle spaces
            if self.add_space_token:
                parts = cleaned_text.split(' ')
                tokens = []
                for i, part in enumerate(parts):
                    tokens.extend(list(part))
                    if i < len(parts) - 1:
                        tokens.append(self.SPACE_TOKEN)
            else:
                tokens = list(cleaned_text)
            
            char_freq.update(tokens)
        
        # Filter by frequency and add to vocabulary
        valid_chars = [
            char for char, freq in char_freq.items()
            if freq >= min_freq and char not in self.char2id
        ]
        
        # Sort for consistency
        valid_chars.sort()
        
        # Add characters
        for char in valid_chars:
            if self.vocab_size is None or len(self.char2id) < self.vocab_size:
                self._add_token(char)
        
        self._vocab_built = True
        
        print(f"Vocabulary built with {len(self.char2id)} tokens")
        print(f"Character frequencies (top 20):")
        for char, freq in char_freq.most_common(20):
            print(f"  '{char}': {freq}")
    
    def _has_header(self, path: str) -> bool:
        with open(path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            return first_line.split('\t')[0].strip().lower == 'path'
    
    def encode(self, text: str, add_blank: bool = False) -> List[int]:
        if not self._vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab_from_metadata() first.")
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Handle spaces
        if self.add_space_token:
            parts = cleaned_text.split(' ')
            tokens = []
            for i, part in enumerate(parts):
                tokens.extend(list(part))
                if i < len(parts) - 1:
                    tokens.append(self.SPACE_TOKEN)
        else:
            tokens = list(cleaned_text)
                    
        # Encode characters
        token_ids = []
        for token in tokens:
            if token in self.char2id:
                token_ids.append(self.char2id[token])
            else:
                token_ids.append(self.unk_id)
        
        # Add blank tokens if requested
        if add_blank:
            token_ids_with_blank = [self.blank_id]
            for token_id in token_ids:
                token_ids_with_blank.append(token_id)
                token_ids_with_blank.append(self.blank_id)
            return token_ids_with_blank
        
        return token_ids
    
    def decode(self, token_ids: List[int], remove_blanks: bool = True) -> str:
        chars = []
        
        for token_id in token_ids:
            # Skip blank tokens
            if remove_blanks and token_id == self.blank_id:
                continue
            
            # Skip padding tokens
            if token_id == self.pad_id:
                continue
            
            # Get character
            if token_id in self.id2char:
                char = self.id2char[token_id]
                
                # Convert space token back to space
                if char == self.SPACE_TOKEN:
                    char = ' '
                
                chars.append(char)
            else:
                chars.append('?')
        
        return ''.join(chars)
    
    def decode_ctc(self, token_ids: List[int]) -> str:
        # Remove consecutive duplicates and blanks
        decoded_ids = []
        prev_id = None
        
        for token_id in token_ids:
            # Skip blank tokens
            if token_id == self.blank_id:
                prev_id = None
                continue
            
            # Skip consecutive duplicates
            if token_id != prev_id:
                decoded_ids.append(token_id)
                prev_id = token_id
        
        return self.decode(decoded_ids, remove_blanks=False)
    
    def vocab_info(self):
        devanagari = [c for c in self.char2id if len(c) == 1 and self._is_devanagari(c)]
        special     = [c for c in self.char2id if c.startswith('<') and c.endswith('>')]
        other       = [c for c in self.char2id if c not in devanagari and c not in special]
        
        print(f"Total vocab size : {len(self.char2id)}")
        print(f"  Special tokens : {len(special)}  {special}")
        print(f"  Devanagari     : {len(devanagari)}")
        print(f"  Other          : {len(other)}  {other}")
        
        def save(self, save_path: str):
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            tokenizer_data = {
                'char2id': self.char2id,
                'id2char': {int(k): v for k, v in self.id2char.items()},
                'vocab_size': self.vocab_size,
                'add_space_token': self.add_space_token,
                'blank_id': self.blank_id,
                'pad_id': self.pad_id,
                'unk_id': self.unk_id,
            }
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
            
            print(f"Tokenizer saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: str) -> 'NepaliTokenizer':
        with open(load_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        tokenizer = cls(
            vocab_size=tokenizer_data['vocab_size'],
            add_space_token=tokenizer_data['add_space_token']
        )
        
        tokenizer.char2id = tokenizer_data['char2id']
        tokenizer.id2char = {int(k): v for k, v in tokenizer_data['id2char'].items()}
        tokenizer.blank_id = tokenizer_data['blank_id']
        tokenizer.pad_id = tokenizer_data['pad_id']
        tokenizer.unk_id = tokenizer_data['unk_id']
        tokenizer._vocab_built = True
        
        print(f"Tokenizer loaded from {load_path}")
        print(f"Vocabulary size: {len(tokenizer.char2id)}")
        
        return tokenizer
    
    def __len__(self) -> int:
        return len(self.char2id)
    
    def get_vocab_size(self) -> int:
        return len(self.char2id)

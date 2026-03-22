import json
import grapheme as glib

def build_lexicon_from_dict(dict_path, output_path, vocab_chars=None):
    with open(dict_path, encoding="utf-8") as f:
        dictionary = json.load(f)

    skip_chars = set(['!', '/', '\u200d', '.', '\u200c', '-', '–', ':', '(', ')', '"', "'"])

    lines = []
    skipped_multi  = 0
    skipped_chars  = 0
    skipped_oov    = 0
    seen_words     = set()

    for entry in dictionary:
        word = entry["word"].strip()

        if " " in word:
            skipped_multi += 1
            continue

        if any(c in skip_chars for c in word):
            skipped_chars += 1
            continue

        if word in seen_words:
            continue
        seen_words.add(word)

        # Split word into grapheme clusters — matches tokenizer behavior
        graphemes = list(glib.graphemes(word))

        # OOV check against grapheme vocab
        if vocab_chars is not None:
            if not all(g in vocab_chars for g in graphemes):
                skipped_oov += 1
                continue

        # Lexicon format: word TAB grapheme1 grapheme2 ...
        char_str = " ".join(graphemes)
        lines.append(f"{word}\t{char_str}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Written:       {len(lines):,} words")
    print(f"Skipped multi: {skipped_multi:,}")
    print(f"Skipped chars: {skipped_chars:,}")
    print(f"Skipped OOV:   {skipped_oov:,}")
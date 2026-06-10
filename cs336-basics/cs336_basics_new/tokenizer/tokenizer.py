import json
import ast
import regex as re
from typing import Iterable, Iterator


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab  
        self.merges = merges  
        self.special_tokens = special_tokens
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.compiled_pat = re.compile(PAT)
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        set_vocab_value = set(vocab.values())
        next_id = len(vocab)
        if special_tokens is not None:
            self.sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
            for special_token in special_tokens:
                if special_token.encode("utf-8") not in set_vocab_value:
                    self.vocab[next_id] = special_token.encode("utf-8")
                    set_vocab_value.add(special_token.encode("utf-8"))
                    next_id += 1
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges_json = json.load(f)
        vocab = {int(v): ast.literal_eval(k) for k, v in vocab_json.items()}
        merges = [(ast.literal_eval(m[0]), ast.literal_eval(m[1])) for m in merges_json]
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        if self.special_tokens is not None:
            pattern = "|".join(re.escape(tok) for tok in self.sorted_special_tokens)
            texts = re.split(f"({pattern})", text)
        else:
            texts = [text]
        
        ids = []
        if self.special_tokens is None:
            set_special_tokens = set()
        else:
            set_special_tokens = set(self.special_tokens)
        for text in texts:
            if text and text in set_special_tokens:
                ids.append(self.inverse_vocab[text.encode("utf-8")])
                continue
            matches = self.compiled_pat.finditer(text)
            tokens = [list(bytes([b]) for b in m.group(0).encode("utf-8")) for m in matches]
            for token in tokens:
                while len(token) > 1:
                    min_rank = len(self.ranks)
                    i = 0
                    while i < len(token) - 1:
                        if (token[i], token[i+1]) in self.ranks:
                            if self.ranks[(token[i], token[i+1])] < min_rank:
                                min_rank = self.ranks[(token[i], token[i+1])]
                        i += 1
                    if min_rank == len(self.ranks):
                        break
                    i = 0
                    while i < len(token) - 1:
                        if self.ranks.get((token[i], token[i+1])) == min_rank:
                            token[i:i+2] = [token[i] + token[i+1]]
                        i += 1
                for tok in token:
                    ids.append(self.inverse_vocab[tok])
        return ids                        
                    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            index = max(buffer.rfind(' '), buffer.rfind('\n'))
            if index != -1:
                yield from self.encode(buffer[:index])
                buffer = buffer[index:]
        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        bytes_list = []
        for i in ids:
            bytes_list.append(self.vocab[i])
        return b''.join(bytes_list).decode("utf-8", errors="replace")

            
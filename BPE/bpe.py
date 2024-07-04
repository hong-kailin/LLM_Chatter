import torch
import torch.nn as nn
import regex as re
import json
from collections import Counter


class BPETokenizer(nn.Module):
    def __init__(self, vocab_path: str, merges_path: str):
        super().__init__()
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        with open(merges_path, "r", encoding="utf-8") as f:
            merges = json.load(f)

        merges = [tuple(value) for key, value in merges.items()]
        self.encoder = vocab
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}
        self.pat = re.compile(r"""
                                 's|'t|'re|'ve|'m|'ll|'d|  # 常见的收缩
                                 \ ?\p{L}+|\ ?\p{N}+|  # 可选空格，后跟1+ unicode字母或数字
                                 \ ?[^\s\p{L}\p{N}]+|  # 可选空格，后面跟着1+非空白/字母/数字
                                 \s+(?!\S)|  # 1+空白字符，后面没有非空白字符
                                 \s+  # 1+空格字符
                                 """, re.X)

    def encode(self, text: str) -> list[int]:
        bbpe_tokens_id = []
        for token in re.findall(self.pat, text):
            bbpe_tokens_id.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token))
        return bbpe_tokens_id

    def decode(self, tokens) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        text = "".join([self.decoder[token] for token in tokens])
        return text

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        chars = [i for i in token]
        for pair in self.bpe_ranks.keys():
            i = 0
            while i < len(chars) - 1:
                if chars[i] == pair[0] and chars[i + 1] == pair[1]:
                    chars = chars[:i] + ["".join(pair)] + chars[i + 2:]
                else:
                    i += 1
        self.cache[token] = chars
        return chars

    @staticmethod
    def train_tokenizer(data,
                        vocab_size,
                        chars_path,
                        vocab_outfile=None,
                        merges_outfile=None):

        if vocab_size < 4625:
            raise ValueError("vocab_size must be greater than 256")

        pat_str = r"'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        split_words = []
        for token in re.findall(pat_str, data):
            split_words.append(token)

        with open(chars_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        vocab = set(data.values())
        merges = []

        while len(vocab) < vocab_size:
            print(len(vocab))
            pair_freq = Counter()

            for split_word in split_words:
                pair_freq.update(zip(split_word[:-1], split_word[1:]))
            most_common_pair = pair_freq.most_common(1)[0][0]

            new_token = most_common_pair[0] + most_common_pair[1]
            vocab.add(new_token)
            merges.append(most_common_pair)

            new_split_words = []
            for split_word in split_words:
                i = 0
                new_word = []
                while i < len(split_word) - 1:
                    if (split_word[i], split_word[i + 1]) == most_common_pair:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(split_word[i])
                        i += 1
                if i == len(split_word) - 1:
                    new_word.append(split_word[i])
                new_split_words.append(new_word)
            split_words = new_split_words
            pass
        vocab = sorted(list(vocab))

        if vocab_outfile:
            with open(vocab_outfile, "w", encoding="utf-8") as f:
                json.dump({v: i for i, v in enumerate(vocab)}, f, ensure_ascii=False)

        char_to_id = {i: char for i, char in enumerate(merges)}
        with open(merges_outfile, 'w', encoding='utf-8') as json_file:
            json.dump(char_to_id, json_file, ensure_ascii=False, indent=4)



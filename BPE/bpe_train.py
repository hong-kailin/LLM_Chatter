from BPE.bpe import BPETokenizer

with open("../BBPE/train_data/斗破苍穹.txt", "r", encoding="utf-8") as f:
    data = f.read()

vocab_size = 4700
vocab_outfile = "vocab.json"
merges_outfile = "merges.json"
BPETokenizer.train_tokenizer(data, vocab_size, "./chars.json", vocab_outfile=vocab_outfile, merges_outfile=merges_outfile)

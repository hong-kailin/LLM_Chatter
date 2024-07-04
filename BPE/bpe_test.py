from bpe import BPETokenizer

tokenizer = BPETokenizer("models/斗破苍穹_bpe/vocab.json", "models/斗破苍穹_bpe/merges.json")

# encode_text = input("请输入要编码的字符串：")
encode_text = "萧炎，你到底是谁？"
print(f"{'#' * 20}编码'{encode_text}'字符串的结果如下{'#' * 20}")

text_id = tokenizer.encode(encode_text)
id_text = tokenizer.decode(text_id)

print(f"'{encode_text}'字符串编码之后对应的id: {text_id}")
print(f"'{encode_text}'字符串编码之后在反编码对应的字符串: {encode_text}")

print('#' * 60)

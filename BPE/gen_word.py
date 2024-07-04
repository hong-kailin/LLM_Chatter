import json

with open(r'C:\Users\1\Desktop\LLM_Chatter\BBPE\train_data\斗破苍穹.txt', 'r', encoding='utf-8') as file:
    text = file.read()

unique_chars = set(text)
char_to_id = {i: char for i, char in enumerate(unique_chars)}
with open('chars.json', 'w', encoding='utf-8') as json_file:
    json.dump(char_to_id, json_file, ensure_ascii=False, indent=4)

print("字符集已保存到 JSON 文件中。")

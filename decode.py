from tokenizers import Tokenizer

# 1. Load the tokenizer file from your resources folder
tokenizer = Tokenizer.from_file("resources/tokenizer.json")

# 2. These are the tokens currently in your main.cpp
tokens = [2, 105, 2364, 107, 155122, 27825, 49087, 531, 496, 236743, 236810, 1051, 2255, 236761, 106, 107, 105, 4368, 107]
tokens += [70895, 1106, 1106, 1106, 1106]
# tokens += [70895, 506, 1902, 563, 1133, 496, 2563]
# 3. Decode to string
output_text = tokenizer.decode(tokens)

print(f"Tokens: {tokens}")
print(f"Decoded Text: {output_text}")
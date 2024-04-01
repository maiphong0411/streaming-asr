from wenet.text.bpe_tokenizer import BpeTokenizer


configs = {
    "bpe_model": "data/lang_char/train_bpe14336.model",
    "symbol_table": 'data/lang_char/train_bpe14336_units.txt',
    "non_lang_syms": None, 
    "split_with_space": False,
}

data_path = "/vinbrain/phongmt/wenet/wenet/examples/foo/s0/text_only.txt"
# initiate bpe model 
tokenizer = BpeTokenizer(configs['bpe_model'], configs['symbol_table'])

with open(data_path, 'r') as file, open('tokens_lm.txt', 'a') as token_file:
    # Read and process each line
    for line in file:
        # Process the line here
        tokens = tokenizer.text2tokens(line)
        result = " ".join(tokens)
        token_file.write(f"{result}\n")
        
# tmp = "bộ tư pháp ban hành quyết định một ba một chín\n"

# result = tokenizer.text2tokens(tmp)
# print(result)
# print("".join(result))
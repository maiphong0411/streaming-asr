from bpe_tokenizer import BpeTokenizer


configs = {
    "bpe_model": "data/lang_char/train_bpe14336.model",
    "symbol_table": 'data/lang_char/train_bpe14336_units.txt',
    "non_lang_syms": None, 
    "split_with_space": False,
}
# initiate bpe model 
tokenizer = BpeTokenizer(configs['bpe_model'], configs['symbol_table'])

tmp = "hello world"

result = tokenizer.text2tokens(tmp)
print(result)
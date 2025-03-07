import json
import torch
import transformer.transformer
import transformer.config
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type = str, help = "enter any file name from 'data' without an extention to load corresponding vocab and model")
    args = parser.parse_args()

    model_path = f'models/{args.file_name}_model.pth'
    vocab_file_path = f'vocabs/{args.file_name}_vocab.json'

    with open(vocab_file_path, 'r', encoding = 'utf8') as f:
        vocab = json.load(f)

    config = transformer.config.config_default
    config.vocab_size = len(vocab)

    max_len = 200
    model = transformer.transformer.Decoder(config)
    model.load_state_dict(torch.load(model_path, 
                                     map_location=torch.device('cpu'), 
                                     weights_only=True))
    model.eval()

    start_token = torch.zeros((1, 1), dtype= torch.int)
    gen_tokens = model.generate(start_token, max_len) 

    decode_tokens = lambda tokens: ''.join([vocab[f'{token}'] for token in tokens])
    gen_text = decode_tokens(gen_tokens[0].tolist())
    print(gen_text)

if __name__ == '__main__':
    main()
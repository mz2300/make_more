import json
import torch
import transformer.transformer
import transformer.config
import argparse

def main():

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type = str, help = "enter any file name from 'data' without an extention to load corresponding vocab and model")
    parser.add_argument("--max_len", type = int, default = 500, help = "num of tokens to generate")
    args = parser.parse_args()


    # model and vocab files
    model_path = f'models/{args.file_name}_model.pth'
    vocab_file_path = f'vocabs/{args.file_name}_vocab.json'


    # load vocab
    with open(vocab_file_path, 'r', encoding = 'utf8') as f:
        vocab_file_d = json.load(f)

        vocab_base = vocab_file_d['initial']
        merges = vocab_file_d['merges']

    # model config
    config = transformer.config.config_default
    config.vocab_size = len(vocab_base) + len(merges)


    # load model    
    model = transformer.transformer.Decoder(config)
    model.load_state_dict(torch.load(model_path, 
                                     map_location=torch.device('cpu'), 
                                     weights_only=True))
    model.eval()


    # generate text
    stoi = {k:v for v, k in vocab_base.items()}
    start_token = torch.tensor([[int(stoi['\n'])]])
    gen_tokens = model.generate(start_token, args.max_len) 

    def unmerge(ids):
        new_ids = []
        for i in ids:
            if i < len(vocab_base):
                new_ids.append(i)
            else:
                merg_pair = merges[i]
                unmerged_ids = unmerge(merg_pair)
                new_ids.extend(unmerged_ids)
        return new_ids

    decode_tokens = lambda tokens: ''.join([vocab_base[f'{token}'] for token in tokens])
    gen_text = decode_tokens(unmerge(gen_tokens[0].tolist()))
    print(gen_text)

if __name__ == '__main__':
    main()
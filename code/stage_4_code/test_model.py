import json

import torch
from code.stage_4_code.Method_RNN_Generator import RNNModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def generate_text(initial_words, model, vocab, idx_to_word, max_length=20):
    model.eval()
    generated_words = initial_words
    for _ in range(max_length):
        inputs = torch.tensor([[vocab.get(word, 0) for word in generated_words[-3:]]], dtype=torch.long, device=device)
        with torch.no_grad():
            output = model(inputs)
        predicted_word_idx = output.argmax(1).item()
        if predicted_word_idx == vocab['<eos>']:
            break
        generated_words.append(idx_to_word[str(predicted_word_idx)])
    return ' '.join(generated_words)


vocab = json.load(open(r'data\stage_4_data\text_generation\loaded_data\vocab.json'))
idx_to_word = json.load(open(r'data\stage_4_data\text_generation\loaded_data\idx_to_word.json'))


model = RNNModel(len(vocab),100, 256, len(vocab)).to(device)
model.load_state_dict(torch.load(r'result\stage_4_result\text_generation\joke_model.pth'))
init_words = ['i', 'like', 'to']
print(generate_text(init_words, model, vocab, idx_to_word))


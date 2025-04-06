from model.tokenization import tokenize
import torch

def predict_next_word(model, vocab, input_text):
    input_tokens = tokenize(input_text, vocab)
    input_tensor = torch.tensor(input_tokens).unsqueeze(0)
    output = model(input_tensor)
    predicted_token = torch.argmax(output[:, -1, :]).item()
    return list(vocab.keys())[list(vocab.values()).index(predicted_token)]
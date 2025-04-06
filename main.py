from train import train_model
from predict import predict_next_word

def main():
    # Train the model
    model, vocab = train_model()
    
    # Make a prediction
    input_text = "hello world how"
    predicted = predict_next_word(model, vocab, input_text)
    print(f"Input: {input_text}, Predicted next word: {predicted}")

if __name__ == "__main__":
    main()
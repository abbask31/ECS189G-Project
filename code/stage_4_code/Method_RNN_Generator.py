import json
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        out = self.fc(output[:, -1, :])  # Get the output of the last time step
        return out


class Method_RNN_Generator:
    data = None
    vocab_path = None
    idx_word_path = None

    def __init__(self, embedding_dim=100, hidden_dim=256, batch_size=128):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = None
        self.idx_to_word = None
        self.model = None


    def _tokenize_sentences(self, sentences):
        return [sentence.lower().split() for sentence in sentences]

    def _build_vocab(self, sentences):
        tokenized_sentences = self._tokenize_sentences(sentences)
        vocabulary = Counter([token for sentence in tokenized_sentences for token in sentence])
        vocab = {word: i + 1 for i, (word, _) in enumerate(vocabulary.items())}
        vocab['<PAD>'] = 0
        return vocab

    def _sentences_to_sequences(self, sentences):
        tokenized_sentences = self._tokenize_sentences(sentences)
        sequences = [[self.vocab[token] for token in sentence] for sentence in tokenized_sentences]
        return sequences

    def create_dataset(self, sequences, sequence_length=15):
        X, y = [], []
        for sequence in sequences:
            for i in range(len(sequence) - sequence_length):
                X.append(sequence[i:i + sequence_length])
                y.append(sequence[i + sequence_length])
        return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def train_model(self, epochs=100):

        self.vocab = self._build_vocab(self.data)
        sequences = self._sentences_to_sequences(self.data)
        vocab_size = len(self.vocab)
        X, y = self.create_dataset(sequences)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = RNNModel(vocab_size, self.embedding_dim, self.hidden_dim, vocab_size).to(self.device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())

        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data, targets) in enumerate(dataloader):
                data = data.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = loss_function(output, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')

        # Save vocab and idx_to_word
        with open(self.vocab_path, 'w') as vocab_f:
            json.dump(self.vocab, vocab_f)

        self.idx_to_word = {i: word for word, i in self.vocab.items()}
        with open(self.idx_word_path, 'w') as idx_to_word_f:
            json.dump(self.idx_to_word, idx_to_word_f)

    def generate_text(self, initial_words, max_length=20):
        if not self.vocab:
            with open('vocab.json', 'r') as vocab_f:
                self.vocab = json.load(vocab_f)
        if not self.idx_to_word:
            with open('idx_to_word.json', 'r') as idx_to_word_f:
                self.idx_to_word = json.load(idx_to_word_f)

        self.model.eval()
        generated_words = initial_words
        for _ in range(max_length):
            inputs = torch.tensor([[self.vocab.get(word, 0) for word in generated_words[-3:]]],
                                  dtype=torch.long, device=self.device)
            with torch.no_grad():
                output = self.model(inputs)
            predicted_word_idx = output.argmax(1).item()
            if self.idx_to_word[predicted_word_idx] == '<eos>':
                break
            generated_words.append(self.idx_to_word[predicted_word_idx])

        return ' '.join(generated_words)


# Example usage
# generator = Method_RNN_Generator()
# generator.train_model()
# initial_words = ['what', 'did', 'the']
# generated_text = generator.generate_text(initial_words)
# generated_text2 = generator.generate_text(['i', 'like', 'chicken'])
# print(generated_text)
# print(generated_text2)



# torch.save(model.state_dict(), r'result\stage_4_result\text_generation\joke_model.pth')
# # Example usage
# initial_words = ['what', 'did', 'the']
# generated_text = generate_text(initial_words, model, vocab, idx_to_word)
# print(generated_text)
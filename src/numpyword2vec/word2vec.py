import numpy as np
from collections import Counter


class Word2Vec:
    def __init__(
        self, dim=50, window=2, num_negative=5, lr=0.025, epochs=5, min_count=1, seed=42
    ):
        self.dim = dim
        self.window = window
        self.num_negative = num_negative
        self.lr = lr
        self.epochs = epochs
        self.min_count = min_count
        self.rng = np.random.default_rng(seed)

        self.word2idx = {}
        self.idx2word = {}

        self.W_in = None
        self.W_out = None
        self.neg_probs = None
        self.vocab_size = 0

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def build_vocab(self, corpus):
        counts = Counter()
        for sent in corpus:
            for w in sent:
                counts[w] += 1

        vocab = []
        for w, c in counts.items():
            if c >= self.min_count:
                vocab.append(w)

        vocab.sort()
        self.vocab_size = len(vocab)

        for i in range(len(vocab)):
            self.word2idx[vocab[i]] = i
            self.idx2word[i] = vocab[i]

        freqs = []
        for i in range(self.vocab_size):
            w = self.idx2word[i]
            freqs.append(counts[w] ** 0.75)

        freqs = np.array(freqs, dtype=np.float64)
        self.neg_probs = freqs / freqs.sum()

    def encode_corpus(self, corpus):
        out = []
        for sent in corpus:
            sent_ids = []
            for w in sent:
                if w in self.word2idx:
                    sent_ids.append(self.word2idx[w])
            if len(sent_ids) > 1:
                out.append(sent_ids)
        return out

    def make_pairs(self, encoded_corpus):
        pairs = []

        for sent in encoded_corpus:
            n = len(sent)

            for i in range(n):
                center = sent[i]

                left = i - self.window
                if left < 0:
                    left = 0

                right = i + self.window + 1
                if right > n:
                    right = n

                for j in range(left, right):
                    if j == i:
                        continue
                    pairs.append((center, sent[j]))

        return pairs

    def sample_negatives(self, pos_id, k):
        negs = []
        while len(negs) < k:
            x = self.rng.choice(self.vocab_size, p=self.neg_probs)
            if x != pos_id:
                negs.append(x)
        return negs

    def init_weights(self):
        self.W_in = self.rng.normal(0, 0.01, size=(self.vocab_size, self.dim))
        self.W_out = np.zeros((self.vocab_size, self.dim))

    def train_pair(self, center_id, context_id):
        v = self.W_in[center_id]
        loss = 0.0

        # positive
        u = self.W_out[context_id]
        s = np.dot(v, u)
        p = Word2Vec.sigmoid(s)

        loss += -np.log(p)

        grad_v = (p - 1.0) * u.copy()
        grad_u = (p - 1.0) * v
        self.W_out[context_id] -= self.lr * grad_u

        # negatives
        neg_ids = self.sample_negatives(context_id, self.num_negative)

        for neg_id in neg_ids:
            u_neg = self.W_out[neg_id]
            s_neg = np.dot(v, u_neg)
            p_neg = Word2Vec.sigmoid(s_neg)

            loss += -np.log(1.0 - p_neg)

            grad_v += p_neg * u_neg.copy()
            self.W_out[neg_id] -= self.lr * (p_neg * v)

        self.W_in[center_id] -= self.lr * grad_v

        return loss

    def fit(self, corpus, verbose=True):
        self.build_vocab(corpus)
        encoded = self.encode_corpus(corpus)
        pairs = self.make_pairs(encoded)
        self.init_weights()

        for epoch in range(self.epochs):
            self.rng.shuffle(pairs)
            total_loss = 0.0

            for center_id, context_id in pairs:
                total_loss += self.train_pair(center_id, context_id)

            if verbose:
                avg_loss = total_loss / max(len(pairs), 1)
                print(f"epoch {epoch + 1}: {avg_loss:.4f}")

        return self

    def get_vector(self, word):
        return self.W_in[self.word2idx[word]]

    def most_similar(self, word, top_k=5):
        v = self.get_vector(word)
        v_norm = np.linalg.norm(v)

        sims = []
        for i in range(self.vocab_size):
            if i == self.word2idx[word]:
                continue

            other = self.W_in[i]
            score = np.dot(v, other) / (v_norm * np.linalg.norm(other))
            sims.append((self.idx2word[i], float(score)))

        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

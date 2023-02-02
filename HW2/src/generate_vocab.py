if __name__ == '__main__':
    train_file = open('train.txt', 'r')
    vocab = {}
    THRESHOLD = 3

    while True:
        line = train_file.readline()

        if not line:
            break

        if line.strip():
            word = line.split("\t")[1].strip()

            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] = vocab[word] + 1

    train_file.close()

    unknown_count = 0
    for w in vocab:
        if vocab[w] < THRESHOLD:
            unknown_count = unknown_count + vocab[w]

    vocab_file = open('vocab.txt', 'w')

    index = 0
    vocab_file.write("<unk>" + "\t" + str(index) + "\t" + str(unknown_count) + "\n")

    for w in sorted(vocab, key=vocab.get, reverse=True):
        if vocab[w] >= THRESHOLD:
            index = index + 1
            vocab_file.write(w + "\t" + str(index) + "\t" + str(vocab[w]) + "\n")

    vocab_file.close()

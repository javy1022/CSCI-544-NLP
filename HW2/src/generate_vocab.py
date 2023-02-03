THRESHOLD = 3


def output_vocab_txt():
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


def dict_add(dict_key, dict_name):
    if dict_key not in dict_name:
        dict_name[dict_key] = 1
    else:
        dict_name[dict_key] = dict_name[dict_key] + 1


def generate_vocab_and_pos_tags_dicts():
    train_file = open('train.txt', 'r')

    previous_pos_tag = ""
    is_first_line = True
    pos_tags_count["START"] = 1
    pos_tags_count["END"] = 1

    while True:
        line = train_file.readline()
        # if eof
        if not line:
            # edge case
            key = previous_pos_tag + " to " + "END"
            dict_add(key, HMM_assum_sequences_count)
            break
        # if line not empty
        if line.strip():
            word = line.split("\t")[1].strip()
            pos_tag = line.split("\t")[2].strip()
            emission_key = pos_tag + " to " + word

            dict_add(word, vocab)
            dict_add(pos_tag, pos_tags_count)
            dict_add(emission_key, pos_tags_to_words_count)
            # add element to HMM dict
            if is_first_line:
                # edge case
                key = "START" + " to " + pos_tag
                HMM_assum_sequences_count[key] = 1
                previous_pos_tag = pos_tag
                is_first_line = False
            else:
                if previous_pos_tag != " ":
                    key = previous_pos_tag + " to " + pos_tag
                    dict_add(key, HMM_assum_sequences_count)
                else:
                    key = "START" + " to " + pos_tag
                    dict_add(key, HMM_assum_sequences_count)
                previous_pos_tag = pos_tag
        # if line is empty
        else:
            pos_tags_count["START"] = pos_tags_count["START"] + 1
            pos_tags_count["END"] = pos_tags_count["END"] + 1
            key = previous_pos_tag + " to " + "END"
            dict_add(key, HMM_assum_sequences_count)
            previous_pos_tag = " "

    train_file.close()

    output_vocab_txt()
    return


def generate_transition_dict():
    transition_temp = {}

    for key, value in HMM_assum_sequences_count.items():
        first_pos_tag = key.split(" ")[0]
        second_pos_tag = key.split(" ")[2]
        transition_key = "(" + first_pos_tag + "," + second_pos_tag + ")"
        transition_temp[transition_key] = value / pos_tags_count[first_pos_tag]

    return transition_temp


def generate_emission_dict():
    emission_temp = {}

    for key, value in pos_tags_to_words_count.items():
        post_tag = key.split(" ")[0]
        word = key.split(" ")[2]
        emission_key = "(" + post_tag + "," + word + ")"
        emission_temp[emission_key] = value / pos_tags_count[post_tag]

    return emission_temp


if __name__ == '__main__':
    vocab = {}
    pos_tags_count = {}
    HMM_assum_sequences_count = {}
    pos_tags_to_words_count = {}

    generate_vocab_and_pos_tags_dicts()
    transition = generate_transition_dict()
    emission = generate_emission_dict()
    
    """
    transition_file = open('transition_debug.txt', 'w')
    transition_file.write(str(transition))
    transition_file.close()
   
    emission_file = open('emission_debug.txt', 'w')
    emission_file.write(str(emission))
    emission_file.close()
    """

import json

TRAIN_FILE = "train.txt"
THRESHOLD = 2
PENN_TREE_BANK_TAGSET = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS",
                         "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$",
                         "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP",
                         "VBZ", "WDT", "WP", "WP$", "WRB", "$", "#", "``", "''", "(", ")", ",", ".", ":"]


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

    print("##### Task 1 #####")
    print("Selected threshold : " + str(THRESHOLD))
    if unknown_count == 0:
        print("Vocabulary total size : " + str(index))
    else:
        print("Vocabulary total size : " + str(index + 1))
    print("Total occurrences of <unk> : " + str(unknown_count) + "\n")

    vocab_file.close()


def dict_add(dict_key, dict_name):
    if dict_key not in dict_name:
        dict_name[dict_key] = 1
    else:
        dict_name[dict_key] = dict_name[dict_key] + 1


def generate_vocab():
    train_file = open(TRAIN_FILE, 'r')
    while True:
        line = train_file.readline()
        if not line:
            break
        if line.strip():
            word = line.split("\t")[1].strip()
            dict_add(word, vocab)
    train_file.close()
    output_vocab_txt()
    return


def generate_pos_tags_dicts():
    train_file = open(TRAIN_FILE, 'r')

    previous_pos_tag = ""
    is_first_line = True
    pos_tags_count["START"] = 1

    while True:
        line = train_file.readline()
        # if eof
        if not line:
            # edge case
            break
        # if line not empty
        if line.strip():
            word = line.split("\t")[1].strip()
            pos_tag = line.split("\t")[2].strip()
            dict_add(pos_tag, pos_tags_count)

            if vocab[word] < THRESHOLD:
                emission_key = pos_tag + " to " + "<unk>"
                dict_add(emission_key, pos_tags_to_words_count)
            else:
                emission_key = pos_tag + " to " + word
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
            previous_pos_tag = " "

    train_file.close()
    return


def generate_transition_dict():
    transition_temp = {}

    for key, value in HMM_assum_sequences_count.items():
        first_pos_tag = key.split(" ")[0]
        second_pos_tag = key.split(" ")[2]
        transition_key = "(" + first_pos_tag + "," + second_pos_tag + ")"
        transition_temp[transition_key] = value / pos_tags_count[first_pos_tag]

    print("##### Task 2 #####")
    print("Total transition parameters : " + str(len(transition_temp)))
    return transition_temp


def generate_emission_dict():
    emission_temp = {}

    for key, value in pos_tags_to_words_count.items():
        post_tag = key.split(" ")[0]
        word = key.split(" ")[2]
        emission_key = "(" + post_tag + "," + word + ")"
        emission_temp[emission_key] = value / pos_tags_count[post_tag]

    print("Total emission parameters : " + str(len(emission_temp)) + "\n")
    return emission_temp


def predict_pos_tag(word, is_first_line, hmm_dicts, previous_predicted_pos_tag):
    highest_prob_pos_tag = [0, "N/A"]

    for pos_tag in PENN_TREE_BANK_TAGSET:

        if is_first_line or previous_predicted_pos_tag == " ":
            transition_key = "(" + "START" + "," + pos_tag + ")"
        else:
            transition_key = "(" + previous_predicted_pos_tag + "," + pos_tag + ")"

        if transition_key not in hmm_dicts[0]:

            continue
        else:
            if word not in vocab:
                emission_key = "(" + pos_tag + "," + "<unk>" + ")"

                if emission_key not in hmm_dicts[1]:
                    continue
                else:
                    emission_prob = hmm_dicts[1][emission_key]
            elif vocab[word] < THRESHOLD:
                emission_key = "(" + pos_tag + "," + "<unk>" + ")"

                if emission_key not in hmm_dicts[1]:
                    continue
                else:
                    emission_prob = hmm_dicts[1][emission_key]

            else:

                emission_key = "(" + pos_tag + "," + word + ")"
                if emission_key not in hmm_dicts[1]:
                    continue
                else:
                    emission_prob = hmm_dicts[1][emission_key]

            transition_prob = hmm_dicts[0][transition_key]
            pos_tag_prob = [transition_prob * emission_prob, pos_tag]

            if pos_tag_prob[0] > highest_prob_pos_tag[0]:
                highest_prob_pos_tag[0] = pos_tag_prob[0]
                highest_prob_pos_tag[1] = pos_tag_prob[1]

    return highest_prob_pos_tag[1]


def greedy_decoding(input_file, hmm_graph, output=False):
    if output:
        output_file = open("greedy.out.txt", 'w')
    else:
        total_words_predicted = 0
        correct_prediction_counts = 0

    # Greedy decoding
    with open(hmm_graph, 'r') as hmm_graph_obj, open(input_file, 'r') as input_file_obj:
        hmm_dicts = json.load(hmm_graph_obj)
        is_first_line = True
        previous_predicted_pos_tag = ""

        while True:
            line = input_file_obj.readline()
            if not line:
                break
            if line.strip():
                if not output:
                    correct_pos_tag = line.split("\t")[2].strip()

                word_to_predict = line.split("\t")[1].strip()

                if is_first_line:
                    predicted_pos_tag = predict_pos_tag(word_to_predict, is_first_line, hmm_dicts,
                                                        previous_predicted_pos_tag)

                    if not output:
                        total_words_predicted = total_words_predicted + 1
                        if predicted_pos_tag == correct_pos_tag:
                            correct_prediction_counts = correct_prediction_counts + 1
                    else:
                        output_line = line + "\t" + predicted_pos_tag
                        output_file.write("\t".join(output_line.split()) + "\n")

                    previous_predicted_pos_tag = predicted_pos_tag
                    is_first_line = False
                else:
                    predicted_pos_tag = predict_pos_tag(word_to_predict, is_first_line, hmm_dicts,
                                                        previous_predicted_pos_tag)

                    if not output:
                        total_words_predicted = total_words_predicted + 1
                        if predicted_pos_tag == correct_pos_tag:
                            correct_prediction_counts = correct_prediction_counts + 1
                    else:
                        output_line = line + "\t" + predicted_pos_tag
                        output_file.write("\t".join(output_line.split()) + "\n")

                    previous_predicted_pos_tag = predicted_pos_tag
            else:
                if output:
                    output_file.write("\n")

                previous_predicted_pos_tag = " "

    if not output:
        print("##### Task 3 #####")
        print(
            "Greedy Decoding Accuracy (" + input_file + ") = " + str(correct_prediction_counts / total_words_predicted))
        print("\n")
    else:
        output_file.close()


def viterbi_decoding(sentence, correct_sequence_list):
   
   
   
   


if __name__ == '__main__':
    vocab = {}
    pos_tags_count = {}
    HMM_assum_sequences_count = {}
    pos_tags_to_words_count = {}

    generate_vocab()
    generate_pos_tags_dicts()
    transition = generate_transition_dict()
    emission = generate_emission_dict()

    with open('hmm.json', 'w') as hmm_file:
        json.dump([transition, emission], hmm_file, indent=4)
    """
    ################## Don't forget to change default pos_tag, current: "N/A" #########################################
    greedy_decoding("dev.txt", "hmm.json")
    greedy_decoding("test.txt", "hmm.json", output=True)
    """

    # Viterbi

    with open("hmm.json", 'r') as hmm_file, open("mini_dev.txt", 'r') as input_file:

        viterbi_input_sentence = ""
        correct_pos_tag_sequence = []

        while True:
            line = input_file.readline()
            if not line:
                break
            if line.strip():
                word = line.split("\t")[1].strip()
                correct_pos_tag = line.split("\t")[2].strip()

                viterbi_input_sentence = viterbi_input_sentence + word + " "
                correct_pos_tag_sequence.append(correct_pos_tag)



            else:
                viterbi_decoding(viterbi_input_sentence,  correct_pos_tag_sequence)
                viterbi_input_sentence = ""
                correct_pos_tag_sequence.clear()


                print("blank")

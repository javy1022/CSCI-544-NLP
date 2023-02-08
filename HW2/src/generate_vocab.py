import json
import numpy as np

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


def greedy_predict_pos_tag(word, is_first_line, hmm_dicts, previous_predicted_pos_tag):
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
                    predicted_pos_tag = greedy_predict_pos_tag(word_to_predict, is_first_line, hmm_dicts,
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
                    predicted_pos_tag = greedy_predict_pos_tag(word_to_predict, is_first_line, hmm_dicts,
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


def init_viterbi_matrix(viterbi_matrix_obj, sentence, hmm_dicts_obj):
    for j in range(len(PENN_TREE_BANK_TAGSET)):
        transition_key = "(" + "START" + "," + PENN_TREE_BANK_TAGSET[j] + ")"
        emission_key = "(" + PENN_TREE_BANK_TAGSET[j] + "," + sentence[0] + ")"

        if transition_key not in hmm_dicts_obj[0]:
            viterbi_matrix_obj[0][j] = 0
            continue

        if sentence[0] not in vocab:
            emission_key = "(" + "START" + "," + "<unk>" + ")"
        elif vocab[sentence[0]] < THRESHOLD:
            emission_key = "(" + "START" + "," + "<unk>" + ")"

        if emission_key not in hmm_dicts_obj[1]:
            viterbi_matrix_obj[0][j] = 0
            continue

        viterbi_matrix_obj[0][j] = hmm_dicts_obj[0][transition_key] * hmm_dicts_obj[1][emission_key]

    return viterbi_matrix_obj


def catch_columns_all_zeros(viterbi_matrix_obj, index_i=None, first_columns=False):
    if first_columns:
        i = 0
    else:
        i = index_i

    all_zero = True
    for j in range(0, len(PENN_TREE_BANK_TAGSET)):
        if viterbi_matrix_obj[i][j] != 0:
            all_zero = False
    if all_zero:
        viterbi_matrix_obj[i] = np.ones(len(PENN_TREE_BANK_TAGSET))
    return viterbi_matrix_obj


def get_optimal_path(viterbi_matrix_obj, backpointer_matrix_obj, optimal_path_obj, sentence_obj):
    optimal_path_prob = 0
    optimal_end_pos_tag_j = -1
    for j in range(len(PENN_TREE_BANK_TAGSET)):
        if viterbi_matrix_obj[len(sentence_obj) - 1][j] > optimal_path_prob:
            optimal_path_prob = viterbi_matrix_obj[len(sentence_obj) - 1][j]
            optimal_end_pos_tag_j = j

    optimal_path_obj.append(PENN_TREE_BANK_TAGSET[optimal_end_pos_tag_j])
    if len(sentence_obj) > 1:
        optimal_previous_pos_tag_j = backpointer_matrix_obj[len(sentence_obj) - 1][optimal_end_pos_tag_j]
        optimal_previos_pos_tag = PENN_TREE_BANK_TAGSET[optimal_previous_pos_tag_j]
        optimal_path_obj.insert(0, optimal_previos_pos_tag)

        for i in reversed(range(1, len(sentence_obj) - 1)):
            optimal_previous_pos_tag_j = backpointer_matrix_obj[i][optimal_previous_pos_tag_j]
            optimal_previos_pos_tag = PENN_TREE_BANK_TAGSET[optimal_previous_pos_tag_j]
            optimal_path_obj.insert(0, optimal_previos_pos_tag)

    return optimal_path_obj


def count_correct_prediction(predicted_pos_tags_list_obj, correct_pos_tag_sequence_obj, counts):
    for i in range(0, len(correct_pos_tag_sequence_obj)):
        if predicted_pos_tags_list_obj[i] == correct_pos_tag_sequence_obj[i]:
            counts = counts + 1
    return counts


def viterbi_decoding(sentence, hmm_dicts_obj):
    viterbi_matrix = np.zeros([len(sentence), len(PENN_TREE_BANK_TAGSET)])
    backpointer_matrix = np.zeros([len(sentence), len(PENN_TREE_BANK_TAGSET)], dtype=int)

    # init viterbi matrix
    viterbi_matrix = init_viterbi_matrix(viterbi_matrix, sentence, hmm_dicts_obj)
    catch_columns_all_zeros(viterbi_matrix, first_columns=True)

    # init backpointer matrix
    for j in range(len(PENN_TREE_BANK_TAGSET)):
        backpointer_matrix[0][j] = 0

    # bottom-up Viterbi algorithm, populate the viterbi matrix & backpointer matrix
    for i in range(1, len(sentence)):
        for j in range(len(PENN_TREE_BANK_TAGSET)):
            highest_current_path_prob = 0
            highest_current_path_prob_pos_tag = 0
            for j2 in range(len(PENN_TREE_BANK_TAGSET)):
                transition_key = "(" + PENN_TREE_BANK_TAGSET[j2] + "," + PENN_TREE_BANK_TAGSET[j] + ")"
                emission_key = "(" + PENN_TREE_BANK_TAGSET[j] + "," + sentence[i] + ")"

                if transition_key not in hmm_dicts_obj[0]:
                    transition_prob = 0
                else:
                    transition_prob = hmm_dicts_obj[0][transition_key]

                if sentence[i] not in vocab:
                    emission_key = "(" + PENN_TREE_BANK_TAGSET[j] + "," + "<unk>" + ")"
                elif vocab[sentence[i]] < THRESHOLD:
                    emission_key = "(" + PENN_TREE_BANK_TAGSET[j] + "," + "<unk>" + ")"

                if emission_key not in hmm_dicts_obj[1]:
                    emission_prob = 0
                else:
                    emission_prob = hmm_dicts_obj[1][emission_key]

                previos_path_prob = viterbi_matrix[i - 1][j2]
                current_path_prob = previos_path_prob * transition_prob * emission_prob
                if current_path_prob > highest_current_path_prob:
                    highest_current_path_prob = current_path_prob
                    highest_current_path_prob_pos_tag = j2

            viterbi_matrix[i][j] = highest_current_path_prob
            backpointer_matrix[i][j] = highest_current_path_prob_pos_tag

        catch_columns_all_zeros(viterbi_matrix, i)

    # backtrace to get optimal path
    optimal_path = []
    optimal_path = get_optimal_path(viterbi_matrix, backpointer_matrix, optimal_path, sentence)
    return optimal_path


def viterbi_decoding_readline_helper():

    with open("hmm.json", 'r') as hmm_graph, open("dev.txt", 'r') as input_file:
        hmm_dicts = json.load(hmm_graph)
        viterbi_input_sentence_list = []
        correct_pos_tag_sequence = []
        total_words_predicted = 0
        correct_prediction_counts = 0

        while True:
            line = input_file.readline()
            if not line:
                predicted_pos_tags_list = viterbi_decoding(viterbi_input_sentence_list,hmm_dicts)
                total_words_predicted = total_words_predicted + len(viterbi_input_sentence_list)
                correct_prediction_counts = count_correct_prediction(predicted_pos_tags_list, correct_pos_tag_sequence,
                                                                     correct_prediction_counts)
                break

            if line.strip():
                word = line.split("\t")[1].strip()
                correct_pos_tag = line.split("\t")[2].strip()
                viterbi_input_sentence_list.append(word)
                correct_pos_tag_sequence.append(correct_pos_tag)
            else:
                predicted_pos_tags_list = viterbi_decoding(viterbi_input_sentence_list,hmm_dicts)
                total_words_predicted = total_words_predicted + len(viterbi_input_sentence_list)
                correct_prediction_counts = count_correct_prediction(predicted_pos_tags_list, correct_pos_tag_sequence,
                                                                     correct_prediction_counts)
                viterbi_input_sentence_list.clear()
                correct_pos_tag_sequence.clear()

        print("##### Task4 ####")
        print("Viterbi Decoding Accuracy : " + str(correct_prediction_counts / total_words_predicted))


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

    ################## Don't forget to change default pos_tag, current: "N/A" #########################################
    greedy_decoding("dev.txt", "hmm.json")
    #greedy_decoding("test.txt", "hmm.json", output=True)


    viterbi_decoding_readline_helper()

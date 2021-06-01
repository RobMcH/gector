from utils.helpers import read_lines


def check_preds(pred_file, gt_file, train_file, correct_pred_file, correct_pred_ori_file):
    """
    :param pred_file: file containing predicted sentences
    :param gt_file: file containing ground truth correct sentences for the predictions
    :param train_file: file containing the original, incorrect sentences correspond to the sentences that were pred
    :param correct_pred_file: file to write correctly predicted sentence to
    :param correct_pred_ori_file: file to write original sentences corresponding to the sentences that were correctly
    predicted
    """
    pred_data = read_lines(pred_file, skip_strip=True)
    gt_data = read_lines(gt_file, skip_strip=True)  # ground truth
    train_data = read_lines(train_file, skip_strip=True)
    pred_sentences = [sent.split() for sent in pred_data]
    gt_sentences = [sent.split() for sent in gt_data]
    train_sentences = [sent.split() for sent in train_data]
    same_count = 0
    correct_sentences = []
    ori_correct_sentences = []
    assert len(gt_sentences) == len(pred_sentences)
    for s1, s2, s3 in zip(gt_sentences, pred_sentences, train_sentences):
        if s1 == s2:  # if prediction = correct label
            correct_sentences.append(s1)  # store correct prediction
            ori_correct_sentences.append(s3)  # store original sentence
            if s1 == s3:
                same_count += 1
    print('same count:', same_count)
    print('num correct sentences:', len(correct_sentences))

    with open(correct_pred_file, 'w') as f:
        f.write("\n".join([" ".join(x) for x in correct_sentences]) + '\n')

    with open(correct_pred_ori_file, 'w') as f:
        f.write("\n".join([" ".join(x) for x in ori_correct_sentences]) + '\n')


def main():
    pred_file = 'model_predictions'
    gt_file = '../GEC-Data/gector_data/wi_dev_cor.txt'
    output_file_1 = 'test_cases/correct_sents_pred.txt'
    output_file_2 = 'test_cases/correct_sents_ori.txt'
    train_file = '../GEC-Data/gector_data/wi_dev_ori.txt'
    check_preds(pred_file, gt_file, train_file, output_file_1, output_file_2)


if __name__ == '__main__':
    main()

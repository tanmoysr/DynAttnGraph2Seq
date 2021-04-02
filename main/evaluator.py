import bleu
import distance
import rouge

def evaluate(type, golds, preds):
    # assert len(golds) == len(preds)
    if type == "acc":
        correct = 0.0
        for _ in range(len(golds)):
            gold = golds[_]
            gold_str = " ".join(gold).strip()

            pred = preds[_]
            pred_str = " ".join(pred).strip()

            if gold_str == pred_str:
                correct += 1.0
        return correct/len(preds)
        
    if type == "bleu":
        score = 0.0
        gold = []
        pred = []
        for seq in golds:
            gold.append([seq.split()])
        for seq in preds:
            pred.append(seq.split())
        # bleu_score, precisions, bp, ratio, translation_length, reference_length
        # bleu_score, _, _, _, _, _ = bleu.compute_bleu(gold, pred, 4, False)
        bleu_score1, _, _, _, _, _ = bleu.compute_bleu(gold, pred, 1, False)
        bleu_score2, _, _, _, _, _ = bleu.compute_bleu(gold, pred, 2, False)
        bleu_score3, _, _, _, _, _ = bleu.compute_bleu(gold, pred, 3, False)
        bleu_score4, _, _, _, _, _ = bleu.compute_bleu(gold, pred, 4, False)
        # bleu_score = 100 * bleu_score
        return bleu_score1, bleu_score2, bleu_score3, bleu_score4

    if type == "rouge":
        score = 0.0
        gold = []
        pred = []
        for seq in golds:
            gold.append(seq.strip())
        for seq in preds:
            pred.append(seq.strip())

        rouge_score_map = rouge.rouge(pred, gold)
        return 100 * rouge_score_map["rouge_1/f_score"]

    if type == "levenshtein":
        score = 0.0
        for _ in range(len(golds)):
            gold = golds[_].split()
            pred = preds[_].split()
            score += distance.levenshtein(gold, pred)

        return score/len(golds)

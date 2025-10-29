from collections import defaultdict, Counter
import math

ALPHABET = [chr(i) for i in range(ord("a"), ord("z") + 1)]
START, END = "<S>", "<E>"

# Read Aspell dictionary file
def read_aspell(path):
    """Reads an Aspell dictionary file and returns a set of words."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            correct, wrong = line.strip().split(":", 1)
            correct = correct.strip().lower()
            wrong = [w.strip().lower() for w in wrong.split()]
            pairs.append((correct, wrong))
    return pairs

# Calculate emission probabilities
def emission(pairs):
    """Builds an emission probability dictionary from word pairs."""
    emit_counts = {c: Counter() for c in ALPHABET}
    for correct, wrong_list in pairs:
        for wrong in wrong_list:
            if len(correct) != len(wrong):
                continue
            for c, w in zip(correct, wrong):
                if c in ALPHABET and w in ALPHABET:
                    emit_counts[c][w] += 1
    emit_probs = {}
    for c in ALPHABET:
        total = sum(emit_counts[c].values())
        # If we have no counts for this correct character, leave its emission dict empty.
        if total == 0:
            emit_probs[c] = {}
        else:
            emit_probs[c] = {w: math.log(count / total) for w, count in emit_counts[c].items()}
    return emit_probs

# Calculate transition probabilities
def transition(pairs):
    """Builds a transition probability dictionary from word pairs."""
    trans_counts = defaultdict(Counter)

    # Count transitions including start and end tokens
    for correct, _ in pairs:
        prev_char = START
        for char in correct:
            if char in ALPHABET:
                trans_counts[prev_char][char] += 1
                prev_char = char
        trans_counts[prev_char][END] += 1

    trans_probs = {}

    # Convert counts to log probabilities
    for prev in trans_counts:
        total = sum(trans_counts[prev].values())
        trans_probs[prev] = {}
        for next in trans_counts[prev]:
            trans_probs[prev][next] = math.log((trans_counts[prev][next] + 1) / (total + len(ALPHABET)))
    return trans_probs

def viterbi_decode(word, emit_probs, trans_probs):
    """Decodes a misspelled word using the Viterbi algorithm."""
    V = [{}]
    path = {}

    for c in ALPHABET:
        emit_prob = emit_probs[c].get(word[0], 0) # emission probability
        trans_prob = trans_probs[START].get(c, float('-inf')) # transition from start to first char
        V[0][c] = trans_prob + emit_prob # total log prob
        path[c] = [c] # initial path

    for t in range(1, len(word)):
        V.append({})
        new_path = {}

        # iterate through all possible current characters
        for c in ALPHABET:
            # calculate max probability and previous state
            emit_prob = emit_probs.get(c, {}).get(word[t], -math.inf)
            # find the best previous character
            (prob, state) = max(
                (V[t - 1][prev_c] + trans_probs.get(prev_c, {}).get(c, float('-inf')) + emit_prob, prev_c)
                for prev_c in ALPHABET
            )
            V[t][c] = prob
            new_path[c] = path[state] + [c]
        path = new_path
    
    # Terminating step
    (prob, state) = max(
        (V[len(word) - 1][c] + trans_probs.get(c, {}).get(END, float('-inf')), c)
        for c in ALPHABET
    )
    return ''.join(path[state])

# Score a correct word against a misspelled word
def viterbi_score(correct, wrong, emit_probs, trans_probs):
    """Scores a correct word against a misspelled word using Viterbi."""
    if len(correct) != len(wrong): # lengths must match
        return float('-inf')
    
    # initialize score with start transition
    score = trans_probs[START].get(correct[0], float('-inf'))

    # iterate through each character position
    for i in range(len(correct)):
        # calculate emission probability for the current character
        emit_prob = emit_probs.get(correct[i], {}).get(wrong[i], -math.inf)
        score += emit_prob
        if i > 0: # add transition probability from previous character
            score += trans_probs.get(correct[i-1], {}).get(correct[i], float('-inf'))
    # add transition probability to end
    score += trans_probs.get(correct[-1], {}).get(END, float('-inf'))
    return score

# Find the best matching correct word for a misspelled word
def correct_word(wrong, correct_words, emit_probs, trans_probs):
    """Finds the best matching correct word for a misspelled word."""
    candidates = [w for w in correct_words if len(w) == len(wrong)]
    if not candidates:
        return wrong
    scores = [(viterbi_score(c, wrong, emit_probs, trans_probs), c) for c in candidates]    
    return max(scores)[1]

# Correct user input text
def correct_text(input_text, emit_probs, trans_probs):
    words = input_text.lower().split()
    corrected_words = []

    for word in words:
        corrected_words.append(correct_word(word, correct_words, emit_probs, trans_probs))
    return ' '.join(corrected_words)

if __name__ == "__main__":
    # Initialize the model with training data
    pairs = read_aspell("aspell.txt")
    emit_probs = emission(pairs)
    trans_probs = transition(pairs)
    correct_words = [p[0] for p in pairs]

    while True: # loop for user input
        text = input("Enter text: ")
        print("Corrected text:", correct_text(text, emit_probs, trans_probs))
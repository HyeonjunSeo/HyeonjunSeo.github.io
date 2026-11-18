from collections import Counter
import numpy as np

def read_documents(doc_file):
    docs = []
    labels = []

    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()

            docs.append(words[3:])   # Remove topic and document ID
            labels.append(words[1])  # pos or neg

    return docs, labels

def train_nb(documents, labels, alpha=1.0):

    classes = set(labels)

    # Count documents belonging to each class
    class_doc_counts = Counter(labels)

    # Count word occurrences for each class
    class_word_counts = {}
    class_total_words = {}

    for c in classes:
        class_word_counts[c] = Counter()
        class_total_words[c] = 0

    vocabulary = set()

    for doc, label in zip(documents, labels):

        class_word_counts[label].update(doc)
        class_total_words[label] += len(doc)
        vocabulary.update(doc)

    # Add one additional vocabulary entry for unknown words
    vocab_size = len(vocabulary) + 1

    # Calculate log prior probabilities
    log_priors = {}

    total_documents = len(documents)

    for c in classes:
        log_priors[c] = np.log(
            class_doc_counts[c] / total_documents
        )

    # Return the model parameters
    model = {
        'classes': classes,
        'log_priors': log_priors,
        'class_word_counts': class_word_counts,
        'class_total_words': class_total_words,
        'vocab_size': vocab_size,
        'alpha': alpha
    }

    return model

def score_doc_label(document, label, model):

    score = model['log_priors'][label]

    word_counts = model['class_word_counts'][label]
    total_words = model['class_total_words'][label]

    alpha = model['alpha']
    vocab_size = model['vocab_size']

    denominator = total_words + alpha * vocab_size

    for word in document:

        numerator = word_counts[word] + alpha

        word_prob = numerator / denominator

        score += np.log(word_prob)

    return score

def classify_nb(document, model):

    scores = {}

    for c in model['classes']:
        scores[c] = score_doc_label(document, c, model)

    return max(scores, key=scores.get)

def classify_documents(docs, model):

    predictions = []

    for doc in docs:
        predictions.append(classify_nb(doc, model))

    return predictions

def accuracy(true_labels, guessed_labels):

    correct = sum(
        true == guessed
        for true, guessed in zip(true_labels, guessed_labels)
    )

    return correct / len(true_labels)

all_docs, all_labels = read_documents(
    'all_sentiment_shuffled.txt'
)

split_point = int(0.80 * len(all_docs))

train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]

eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]

model = train_nb(train_docs, train_labels, alpha=1.0)

test_docs = [
    ['great'],
    ['bad'],
    ['a', 'top-quality', 'performance']
]

for doc in test_docs:

    pos_score = score_doc_label(doc, 'pos', model)
    neg_score = score_doc_label(doc, 'neg', model)

    print("\nDocument:", doc)
    print("Positive score:", pos_score)
    print("Negative score:", neg_score)

    print("Positive probability:", np.exp(pos_score))
    print("Negative probability:", np.exp(neg_score))

    print("Prediction:", classify_nb(doc, model))

predictions = classify_documents(eval_docs, model)

eval_accuracy = accuracy(eval_labels, predictions)

print("\nEvaluation accuracy:", eval_accuracy)

misclassified = []

for doc, true_label, predicted_label in zip(
    eval_docs, eval_labels, predictions
):

    if true_label != predicted_label:

        pos_score = score_doc_label(doc, 'pos', model)
        neg_score = score_doc_label(doc, 'neg', model)

        misclassified.append({
            'document': doc,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'pos_score': pos_score,
            'neg_score': neg_score
        })


print("\nNumber of misclassified documents:", len(misclassified))

misclassified.sort(key=lambda x: len(x["document"]))

shortest_misclassified = misclassified[:10]

print("\nERROR ANALYSIS")
print("=" * 50)
print(f"Number of misclassified documents: {len(misclassified)}")

for i, item in enumerate(shortest_misclassified, 1):
    print(f"\nMisclassified review {i}")
    print(f"Review: {" ".join(item['document'])}")
    print(f"True label: {item['true_label']}")
    print(f"Predicted label: {item['predicted_label']}")
    print(f"Positive score: {item['pos_score']}")
    print(f"Negative score: {item['neg_score']}")
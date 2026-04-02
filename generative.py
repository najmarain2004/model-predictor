def evaluate_generative(scores):
    total = sum(scores)
    return {
        "Semantic Accuracy": total / len(scores)
    }

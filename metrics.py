from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_model(y_true, y_pred, label):
    # calculate metrics for model evaluation
    conf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # print the metrics
    print(f'{label} Confusion Matrix:\n', conf_matrix)
    print(f'{label} Accuracy: {accuracy:.4f}')
    print(f'{label} Precision: {precision:.4f}')
    print(f'{label} Recall: {recall:.4f}')
    print(f'{label} F1 Score: {f1:.4f}')

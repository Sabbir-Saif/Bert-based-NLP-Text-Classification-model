from datasets import DatasetDict, Dataset, load_dataset
from keras.src.metrics.accuracy_metrics import accuracy

from tensorflow.python.keras.backend import learning_phase
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from transformers import DataCollatorWithPadding


dataset_dict= load_dataset("shawhin/phishing-site-classification")

model_path="google-bert/bert-base-uncased"
tokenizer= AutoTokenizer.from_pretrained(model_path)
id2label={0: "Safe", 1:"Not Safe"}
label2id= {"Safe":0 , "Not Safe": 1}
model= AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, id2label=id2label, label2id=label2id,)

for name, param in model.base_model.named_parameters():
    if "pooler" in name:
        param.requires_grad = True

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_data=dataset_dict.map(preprocess_function, batched = True)

data_collator= DataCollatorWithPadding(tokenizer=tokenizer)

accuracy= evaluate.load("accuracy")
auc_score = evaluate.load("roc_auc")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    probabilities= np.exp(predictions)/ np.exp(predictions).sum(-1, keepdims=True)
    positive_class_probs= probabilities[:,1]
    auc=np.round(auc_score.compute(prediction_scores=positive_class_probs,references=labels)['roc_auc'],3)
    predicted_classes= np.argmax(predictions, axis=1)
    acc=np.round(accuracy.compute(predictions=predicted_classes,references=labels)['accuracy'],3)
    return {"Accuracy":acc, "AUC":auc}

lr=2e-4
batch_size=8
num_epochs=10

training_args= TrainingArguments(
    output_dir="bert-phishing-classifier_teacher",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

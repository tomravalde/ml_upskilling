import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric

"""
This is an LLM approach, which uses BERT to classify film reviews. It differs from RNN approach:

RNNs do take account of context by sequential processing, with word-by-word updating
However BERT accounts fo global context, via the Transformer architecture, understanding 
bi-directional context and has been pre-trained on a large corpus of text.

i.e. The RNN has no pre-trained weights, but trains from scratch, learning weights through 
backpropogation

BERT, trained on a large corpus of text, can be fine-tuned on specific tasks with smaller 
datasets to updated the weights with task-specific labelled data; generally faster than and RNN, 
though requires a more complicated taokensation process, e.g. to define 'classification' and 
'separation' tokens

As well as classification, can be used for named entity  recognition and question answering

"""

# Load the IMDb dataset
dataset = load_dataset('imdb')

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the input data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare the datasets for training
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
tokenized_datasets.set_format('torch')

# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='../../results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Load the evaluation metric
metric = load_metric('accuracy')

# Define the compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    return metric.compute(predictions=predictions, references=labels)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)
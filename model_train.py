import kagglehub
import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import (DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback, logging as transformers_logging)
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import pickle
import sys
import seaborn as sns
import matplotlib.pyplot as plt


# Root logger
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to reduce clutter; change to DEBUG for more details
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

transformers_logging.set_verbosity_info()
transformers_logging.enable_explicit_format()

logger = logging.getLogger(__name__)


# Dataset Acquisition

logger.info("Starting dataset download...")

try:
    dataset_path = kagglehub.dataset_download("zeyadkhalid/mbti-personality-types-500-dataset")
    logger.info(f"Dataset downloaded to: {dataset_path}")
except Exception as e:
    logger.exception("Failed to download dataset.")
    sys.exit(1)

csv_path = os.path.join(dataset_path, "MBTI 500.csv")

# 
# Data Preprocessing
#

logger.info("Loading dataset...")
try:
    data = pd.read_csv(csv_path)
    logger.info(f"Dataset shape: {data.shape}")
    logger.debug(f"First few rows:\n{data.head()}")
except Exception as e:
    logger.exception("Failed to load dataset.")
    sys.exit(1)

logger.info("Starting data preprocessing...")

# Drop rows with missing values in posts or type
initial_shape = data.shape
data = data.dropna(subset=['posts', 'type'])
logger.info(f"Dropped {initial_shape[0] - data.shape[0]} rows with missing values.")
logger.info(f"Dataset shape after dropping missing values: {data.shape}")

# Features and labels
X = data['posts'].astype(str)
y = data['type']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_labels = len(le.classes_)
logger.info(f"Number of unique MBTI types: {num_labels}")
logger.debug(f"Classes: {le.classes_}")

# Save the label encoder for later use
try:
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    logger.info("Label encoder saved as 'label_encoder.pkl'.")
except Exception as e:
    logger.exception("Failed to save label encoder.")
    sys.exit(1)

# Analyze class distribution
class_counts = data['type'].value_counts()
logger.info(f"Class distribution:\n{class_counts}")

# 
# Data Splitting
#

logger.info("Splitting data into training, validation, and testing sets...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y_encoded,
    test_size=0.3,  # 70% training, 30% temp
    random_state=42,
    stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,  # 15% validation, 15% testing
    random_state=42,
    stratify=y_temp
)

logger.info(f"Training samples: {len(X_train)}")
logger.info(f"Validation samples: {len(X_val)}")
logger.info(f"Testing samples: {len(X_test)}")

#
# Tokenization
#

MAX_LENGTH = 128  # max_length global variable

logger.info("Initializing the tokenizer...")
try:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    logger.info("Tokenizer loaded successfully.")
except Exception as e:
    logger.exception("Failed to load tokenizer.")
    sys.exit(1)

# Define a function to tokenize the texts
def tokenize_texts(texts, split_name=""):
    logger.debug(f"Tokenizing {len(texts)} samples for {split_name} set...")
    return tokenizer(
        texts.tolist(),
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )

# Tokenize training, validation, and testing data
try:
    logger.info("Tokenizing training data...")
    train_encodings = tokenize_texts(X_train, "training")

    logger.info("Tokenizing validation data...")
    val_encodings = tokenize_texts(X_val, "validation")

    logger.info("Tokenizing testing data...")
    test_encodings = tokenize_texts(X_test, "testing")

    logger.info("Tokenization completed.")
except Exception as e:
    logger.exception("Tokenization failed.")
    sys.exit(1)

#
# Handling Class Imbalance
#

logger.info("Computing class weights to handle class imbalance...")
try:
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    logger.info(f"Class weights: {class_weights}")
except Exception as e:
    logger.exception("Failed to compute class weights.")
    sys.exit(1)

#
# 7. Dataset Creation
#

logger.info("Creating Torch datasets...")

class MBTIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
    def __len__(self):
        return len(self.labels)

try:
    train_dataset = MBTIDataset(train_encodings, y_train)
    val_dataset = MBTIDataset(val_encodings, y_val)
    test_dataset = MBTIDataset(test_encodings, y_test)
    logger.info("Torch datasets created successfully.")
except Exception as e:
    logger.exception("Failed to create Torch datasets.")
    sys.exit(1)

#
# Model Initialization
#

logger.info("Initializing the DistilBERT model for sequence classification...")
try:
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=num_labels
    )
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.exception("Failed to load the model.")
    sys.exit(1)

# CUDA
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
logger.info(f"Model is using device: {device}")

#
# Training Arguments
#

logger.info("Setting up training arguments...")

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,                    # Total number of training epochs
    per_device_train_batch_size=16,        # Batch size per device during training
    per_device_eval_batch_size=64,         # Batch size for evaluation
    warmup_steps=100,                     # Reduced number of warmup steps
    weight_decay=0.01,                     # Strength of weight decay
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",           # Evaluate every eval_steps
    eval_steps=50,                         # Number of steps between evaluations
    save_strategy="steps",                 # Save checkpoints every save_steps
    save_steps=100,                        # Number of steps between saves
    save_total_limit=5,                    # Only last 5 checkpoints are saved
    load_best_model_at_end=True,           # Load the best model when finished training
    metric_for_best_model="f1_weighted",   # Use weighted F1-score to evaluate the best model
    greater_is_better=True,
    fp16=torch.cuda.is_available(),        # Enable mixed precision if CUDA is available
    report_to=["none"],                    # Disable reporting to WandB or other services
)

logger.debug(f"Training arguments: {training_args}")

# 
# Metrics Computation
#

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    
    metrics = {
        'accuracy': acc,
        'f1_weighted': f1,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_macro': macro_f1,
        'precision_macro': macro_precision,
        'recall_macro': macro_recall,
    }
    logger.debug(f"Computed metrics: {metrics}")
    return metrics

#
# Trainer Initialization
# 

logger.info("Initializing the Trainer with Early Stopping...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

#
# Training the Model
# 
logger.info("Starting training...")

try:
    trainer.train()
    logger.info("Training completed successfully.")
except Exception as e:
    logger.exception("An error occurred during training.")
    sys.exit(1)

#
# Evaluating the Model
#

logger.info("Evaluating the model on the validation set...")
try:
    val_results = trainer.evaluate()
    logger.info(f"Validation Evaluation results: {val_results}")
except Exception as e:
    logger.exception("An error occurred during validation.")
    sys.exit(1)

#
# Testing and Classification Report
#

logger.info("Generating classification report on the test set...")
try:
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    
    # Decode labels
    y_true = le.inverse_transform(y_test)
    y_pred = le.inverse_transform(preds)
    
    # Print
    report = classification_report(y_true, y_pred)
    logger.info(f"Classification Report:\n{report}")
    
    # Save classification report
    with open("classification_report.txt", "w") as f:
        f.write(report)
    logger.info("Classification report saved as 'classification_report.txt'.")
    
    # Generate
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    logger.info("Confusion matrix saved as 'confusion_matrix.png'.")
except Exception as e:
    logger.exception("An error occurred while generating the classification report.")
    sys.exit(1)

#
# Save the Model and Tokenizer
# 

model_save_path = './mbti-distilbert-model'
logger.info(f"Saving the model and tokenizer to '{model_save_path}'...")
try:
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info("Model and tokenizer saved successfully.")
except Exception as e:
    logger.exception("Failed to save the model and tokenizer.")
    sys.exit(1)
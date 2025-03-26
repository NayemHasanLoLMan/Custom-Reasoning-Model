import os
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader, random_split

# Configuration
MAX_LEN = 150
BATCH_SIZE = 8
VAL_RATIO = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create necessary directories
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 🔹 **Load Dataset**
print("Loading facebook/natural_reasoning dataset...")
dataset = load_dataset("facebook/natural_reasoning")

# 🔹 **Check Available Keys**
print("Dataset columns:", dataset["train"].column_names)  # Debugging step

# 🔹 **Process Dataset Safely**
def process_example(example):
    input_text = f"Question: {example['question']}"

    # Ensure reference answer exists
    output_text = f"Answer: {example.get('reference_answer', 'No reference answer provided.')}"
    
    return {
        "input": input_text,
        "output": output_text,
        "question_id": example.get("question_id", "unknown")  # Avoid KeyError
    }

# 🔹 **Apply Processing**
print("Processing dataset...")
processed_dataset = dataset["train"].map(process_example, remove_columns=dataset["train"].column_names)

# 🔹 **Filter out short/empty answers**
filtered_dataset = processed_dataset.filter(lambda x: len(x["output"]) > 15)
print(f"Dataset size after filtering: {len(filtered_dataset)}")

# 🔹 **Split dataset into Train/Validation**
train_size = int((1 - VAL_RATIO) * len(filtered_dataset))
val_size = len(filtered_dataset) - train_size
train_dataset, val_dataset = random_split(filtered_dataset, [train_size, val_size])
print(f"Train size: {train_size}, Validation size: {val_size}")

# 🔹 **Load Tokenizer & BERT**
print("Loading BERT tokenizer & model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# 🔹 **Dataset Class**
class ReasoningDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=MAX_LEN):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        # Tokenize input
        input_encoding = self.tokenizer(
            example["input"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize output
        output_encoding = self.tokenizer(
            example["output"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "decoder_input_ids": output_encoding["input_ids"].squeeze(0),
            "decoder_attention_mask": output_encoding["attention_mask"].squeeze(0),
            "labels": output_encoding["input_ids"].squeeze(0),
            "question_id": example.get("question_id", "unknown")
        }

# 🔹 **Create Datasets & DataLoaders**
train_dataset_obj = ReasoningDataset(train_dataset, tokenizer)
val_dataset_obj = ReasoningDataset(val_dataset, tokenizer)

train_dataloader = DataLoader(train_dataset_obj, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset_obj, batch_size=BATCH_SIZE, shuffle=False)

print("Dataset processing complete! ✅")

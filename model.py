import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, List, Tuple, Any

# Configuration
MAX_LEN = 150
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
EPOCHS = 5
LEARNING_RATE = 5e-5
EMBED_SIZE = 768  # BERT embeddings
HIDDEN_SIZE = 512
BEAM_SIZE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_RATIO = 0.1
CHECKPOINT_DIR = "./checkpoints"
TENSORBOARD_DIR = "./logs"
DROPOUT = 0.2
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 3
SAVE_INTERVAL = 1
WARMUP_STEPS = 1000

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

# Initialize TensorBoard
writer = SummaryWriter(TENSORBOARD_DIR)

# Load Dataset
print("Loading facebook/natural_reasoning dataset...")
try:
    dataset = load_dataset("facebook/natural_reasoning")
    
    # Examine the dataset structure
    print("Dataset structure:")
    print(f"Columns: {dataset['train'].column_names}")
    example = dataset['train'][0]
    print("Example keys:", list(example.keys()))
    
    # Process dataset
    print("Processing dataset...")
    def process_example(example):
        # Format input and output
        input_text = f"Question: {example['question']}"
        
        # Use reference_answer if available, otherwise use a placeholder
        output_text = f"Answer: {example['reference_answer']}" if example['reference_answer'] else "Answer: No reference answer provided."
        
        return {
            "input": input_text,
            "output": output_text,
            # Don't include question_id if it doesn't exist
        }
    
    # Apply processing to train split
    processed_dataset = dataset["train"].map(process_example, remove_columns=dataset["train"].column_names)
    
    # Filter out short/empty outputs
    filtered_dataset = processed_dataset.filter(lambda x: len(x["output"]) > 15)
    print(f"Dataset size after filtering: {len(filtered_dataset)}")

except Exception as e:
    print(f"Error loading or processing dataset: {e}")
    print("Falling back to SQuAD dataset for demonstration...")
    
    # Fallback to SQuAD dataset
    dataset = load_dataset("squad", split="train[:20000]")
    
    def process_squad(example):
        return {
            "input": f"Question: {example['question']}",
            "output": f"Answer: {example['answers']['text'][0] if example['answers']['text'] else 'No answer provided.'}"
        }
    
    processed_dataset = dataset.map(process_squad)
    filtered_dataset = processed_dataset.filter(lambda x: len(x["output"]) > 15)
    print(f"SQuAD dataset size after filtering: {len(filtered_dataset)}")

# Split dataset
train_size = int((1 - VAL_RATIO) * len(filtered_dataset))
val_size = len(filtered_dataset) - train_size
train_dataset, val_dataset = random_split(filtered_dataset, [train_size, val_size])
print(f"Train size: {train_size}, Validation size: {val_size}")

# Load Tokenizer & Pretrained BERT
print("Loading BERT tokenizer & model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Add special tokens for generation
ANSWER_TOKEN = "[ANS]"
special_tokens = {"additional_special_tokens": [ANSWER_TOKEN]}
tokenizer.add_special_tokens(special_tokens)
bert_model.resize_token_embeddings(len(tokenizer))

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
            "labels": output_encoding["input_ids"].squeeze(0)
        }

# Create datasets and dataloaders
train_dataset_obj = ReasoningDataset(train_dataset, tokenizer)
val_dataset_obj = ReasoningDataset(val_dataset, tokenizer)

train_dataloader = DataLoader(train_dataset_obj, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset_obj, batch_size=BATCH_SIZE, shuffle=False)

# Encoder-Decoder with Attention
# Encoder-Decoder with Attention
# Encoder-Decoder with Attention
class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout):
        super().__init__()
        self.bert = bert_model
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze BERT parameters
            
        # Encoder (BERT + BiGRU)
        self.encoder_rnn = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)

        # Decoder
        self.decoder_embedding = nn.Embedding(vocab_size, embed_size)
        self.decoder_rnn = nn.GRU(embed_size + hidden_size, hidden_size, batch_first=True)  # 768 + 512 = 1280

        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2 + hidden_size, hidden_size)  # 1024 + 512 = 1536 -> 512
        self.attention_combine = nn.Linear(hidden_size, 1)  # Ensures correct shape
        self.context_projection = nn.Linear(hidden_size * 2, hidden_size)  # 1024 -> 512
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size * 2, vocab_size)  # 512 + 512 = 1024

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, teacher_forcing_ratio=0.5):
        batch_size = input_ids.size(0)
        
        # BERT embeddings
        with torch.no_grad():
            bert_output = self.bert(input_ids, attention_mask=attention_mask)[0]  # Shape (batch_size, seq_len, 768)
        
        # Run encoder GRU over BERT embeddings
        encoder_outputs, hidden = self.encoder_rnn(self.dropout(bert_output))  # Outputs (batch_size, seq_len, 1024)

        # Ensure hidden state is correctly reshaped for bidirectional GRU
        hidden = hidden.view(2, batch_size, -1).sum(0, keepdim=True)  # Merge bidirectional states

        # Inference mode (if no decoder input provided)
        if decoder_input_ids is None:
            return self.generate(encoder_outputs, hidden, attention_mask)

        # Teacher Forcing
        decoder_outputs = []
        decoder_input = decoder_input_ids[:, 0].unsqueeze(1)  # Start token

        seq_len = decoder_input_ids.size(1)
        for t in range(1, seq_len):
            # Decoder embedding
            embedded = self.decoder_embedding(decoder_input)  # (batch_size, 1, 768)

            # Attention mechanism
            attention_scores = []
            for i in range(batch_size):
                valid_length = attention_mask[i].sum().item()
                if valid_length > 0:
                    encoder_valid = encoder_outputs[i, :valid_length]  # (valid_length, 1024)
                    decoder_hidden_expanded = hidden[0, i].unsqueeze(0).expand(valid_length, -1)  # (valid_length, 512)
                    concat = torch.cat((encoder_valid, decoder_hidden_expanded), dim=1)  # (valid_length, 1536)

                    # Apply attention
                    attention_hidden = self.attention(concat)  # (valid_length, 512)
                    score = self.attention_combine(attention_hidden).squeeze(-1)  # (valid_length,)
                    attn_weights = F.softmax(score, dim=0)  # (valid_length,)
                    context = torch.matmul(attn_weights.unsqueeze(0), encoder_valid).squeeze(0)  # (1024,)
                    context = self.context_projection(context)  # (512,)
                else:
                    context = torch.zeros(HIDDEN_SIZE).to(DEVICE)  # (512,)
                attention_scores.append(context)

            context_vector = torch.stack(attention_scores).unsqueeze(1)  # (batch_size, 1, 512)

            # Concatenate embedding and context for decoder input
            rnn_input = torch.cat((embedded, context_vector), dim=2)  # (batch_size, 1, 768 + 512 = 1280)

            # Run decoder GRU for one step
            output, hidden = self.decoder_rnn(rnn_input, hidden)  # (batch_size, 1, 512)

            # Generate predictions
            output_context = torch.cat((output.squeeze(1), context_vector.squeeze(1)), dim=1)  # (batch_size, 1024)
            prediction = self.output_layer(self.dropout(output_context))  # (batch_size, vocab_size)

            decoder_outputs.append(prediction)

            # Teacher Forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            if teacher_force:
                decoder_input = decoder_input_ids[:, t].unsqueeze(1)
            else:
                decoder_input = prediction.argmax(1).unsqueeze(1)  # Use predicted token

        # Stack outputs
        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        return decoder_outputs

    def generate(self, encoder_outputs, hidden, attention_mask, max_length=MAX_LEN):
        batch_size = encoder_outputs.size(0)
        
        # Start with BOS token
        decoder_input = torch.full((batch_size, 1), tokenizer.cls_token_id).to(DEVICE)
        
        # Reshape hidden state
        hidden = hidden.view(2, batch_size, -1).sum(0, keepdim=True)
        
        all_tokens = []
        for _ in range(max_length):
            embedded = self.decoder_embedding(decoder_input)  # (batch_size, 1, 768)
            
            # Calculate attention weights
            attention_scores = []
            for i in range(batch_size):
                valid_length = attention_mask[i].sum().item()
                if valid_length > 0:
                    encoder_valid = encoder_outputs[i, :valid_length]  # (valid_length, 1024)
                    decoder_hidden_expanded = hidden[0, i].unsqueeze(0).expand(valid_length, -1)  # (valid_length, 512)
                    concat = torch.cat((encoder_valid, decoder_hidden_expanded), dim=1)  # (valid_length, 1536)
                    attention_hidden = self.attention(concat)  # (valid_length, 512)
                    score = self.attention_combine(attention_hidden).squeeze(-1)  # (valid_length,)
                    attn_weights = F.softmax(score, dim=0)  # (valid_length,)
                    context = torch.matmul(attn_weights.unsqueeze(0), encoder_valid).squeeze(0)  # (1024,)
                    context = self.context_projection(context)  # (512,)
                else:
                    context = torch.zeros(HIDDEN_SIZE).to(DEVICE)  # (512,)
                attention_scores.append(context)
            
            context_vector = torch.stack(attention_scores).unsqueeze(1)  # (batch_size, 1, 512)
            
            # Decoder input
            rnn_input = torch.cat((embedded, context_vector), dim=2)  # (batch_size, 1, 1280)
            output, hidden = self.decoder_rnn(rnn_input, hidden)  # (batch_size, 1, 512)
            
            # Generate predictions
            output_context = torch.cat((output.squeeze(1), context_vector.squeeze(1)), dim=1)  # (batch_size, 1024)
            prediction = self.output_layer(output_context)  # (batch_size, vocab_size)
            
            # Get predicted token
            tokens = prediction.argmax(1).unsqueeze(1)
            all_tokens.append(tokens)
            
            decoder_input = tokens
            
            # Stop if all sequences have reached EOS
            if all((tokens == tokenizer.sep_token_id).any(dim=1)):
                break
                
        # Concatenate all tokens
        all_tokens = torch.cat(all_tokens, dim=1)
        return all_tokens

# Evaluation metrics setup
rouge_calculator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
smooth = SmoothingFunction().method1

def calculate_metrics(references, hypotheses):
    # BLEU score
    bleu_score = corpus_bleu([[ref.split()] for ref in references], [hyp.split() for hyp in hypotheses], smoothing_function=smooth)
    
    # ROUGE scores
    rouge_scores = {metric: 0.0 for metric in ['rouge1', 'rouge2', 'rougeL']}
    for ref, hyp in zip(references, hypotheses):
        scores = rouge_calculator.score(ref, hyp)
        for metric in rouge_scores:
            rouge_scores[metric] += scores[metric].fmeasure
    
    # Average ROUGE scores
    for metric in rouge_scores:
        rouge_scores[metric] /= len(references)
    
    return {
        'bleu': bleu_score,
        **rouge_scores
    }

# Training function
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs=EPOCHS):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            decoder_input_ids = batch["decoder_input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, decoder_input_ids)
            
            # Calculate loss
            loss = criterion(outputs.reshape(-1, len(tokenizer)), labels[:, 1:].reshape(-1))
            loss = loss / ACCUMULATION_STEPS
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or batch_idx == len(train_dataloader) - 1:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
            train_loss += loss.item() * ACCUMULATION_STEPS
            progress_bar.set_postfix({"loss": loss.item() * ACCUMULATION_STEPS})
            
            # Log to TensorBoard
            writer.add_scalar('Training/BatchLoss', loss.item() * ACCUMULATION_STEPS, global_step)
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_dataloader)
        writer.add_scalar('Training/EpochLoss', avg_train_loss, epoch)
        
        # Validation
        model.eval()
        val_loss = 0
        references = []
        hypotheses = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                
                # Generate answers
                outputs = model(input_ids, attention_mask)
                
                # Calculate validation loss
                loss = criterion(
                    model(input_ids, attention_mask, batch["decoder_input_ids"].to(DEVICE)).reshape(-1, len(tokenizer)),
                    labels[:, 1:].reshape(-1)
                )
                val_loss += loss.item()
                
                # Collect references and hypotheses for metrics
                for i in range(len(input_ids)):
                    ref = tokenizer.decode(batch["labels"][i], skip_special_tokens=True)
                    hyp = tokenizer.decode(outputs[i], skip_special_tokens=True)
                    references.append(ref)
                    hypotheses.append(hyp)
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_dataloader)
        writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
        
        # Calculate and log metrics
        metrics = calculate_metrics(references, hypotheses)
        for metric, value in metrics.items():
            writer.add_scalar(f'Validation/{metric}', value, epoch)
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Metrics: BLEU: {metrics['bleu']:.4f}, ROUGE-1: {metrics['rouge1']:.4f}, ROUGE-L: {metrics['rougeL']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % SAVE_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
            }, f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch+1}.pt")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
            }, f"{CHECKPOINT_DIR}/best_model.pt")
            print(f"New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return model

# Beam search implementation
def beam_search(model, input_ids, attention_mask, beam_size=BEAM_SIZE, max_length=MAX_LEN):
    model.eval()
    batch_size = input_ids.size(0)
    
    # Get encoder outputs
    with torch.no_grad():
        bert_output = model.bert(input_ids, attention_mask=attention_mask)[0]
        encoder_outputs, hidden = model.encoder_rnn(bert_output)
    
    # Reshape hidden state
    hidden = hidden.view(2, 2, batch_size, -1)
    hidden = hidden.mean(0).sum(0, keepdim=True)
    
    # Start with start token
    start_token = torch.full((batch_size, 1), tokenizer.cls_token_id).to(DEVICE)
    
    # Initialize beams for each item in batch
    beams = [(start_token, 0.0, hidden)]
    
    for _ in range(max_length):
        candidates = []
        
        # Expand each beam
        for seq, score, hidden in beams:
            # Stop if sequence ended
            if seq[:, -1].item() == tokenizer.sep_token_id:
                candidates.append((seq, score, hidden))
                continue
                
            # Get embedding for last token
            embedded = model.decoder_embedding(seq[:, -1].unsqueeze(1))
            
            # Calculate attention
            attention_scores = []
            for i in range(batch_size):
                valid_length = attention_mask[i].sum().item()
                if valid_length > 0:
                    encoder_valid = encoder_outputs[i, :valid_length]
                    decoder_hidden_expanded = hidden[0, i].unsqueeze(0).expand(valid_length, -1)
                    concat = torch.cat((encoder_valid, decoder_hidden_expanded), dim=1)
                    score_i = model.attention(concat).squeeze(-1)
                    attn_weights = F.softmax(score_i, dim=0)
                    context = torch.matmul(attn_weights.unsqueeze(0), encoder_valid).squeeze(0)
                else:
                    context = torch.zeros(HIDDEN_SIZE*2).to(DEVICE)
                attention_scores.append(context)
            
            context_vector = torch.stack(attention_scores).unsqueeze(1)
            
            # Decoder input
            rnn_input = torch.cat((embedded, context_vector), dim=2)
            output, new_hidden = model.decoder_rnn(rnn_input, hidden)
            
            # Prediction
            output_context = torch.cat((output, context_vector), dim=2)
            prediction = model.output_layer(output_context.squeeze(1))
            
            # Get top k tokens
            log_probs = F.log_softmax(prediction, dim=1)
            topk_probs, topk_ids = log_probs.topk(beam_size)
            
            for j in range(beam_size):
                token_id = topk_ids[0, j].unsqueeze(0).unsqueeze(1)
                token_score = topk_probs[0, j].item()
                new_seq = torch.cat([seq, token_id], dim=1)
                new_score = score + token_score
                candidates.append((new_seq, new_score, new_hidden))
        
        # Select top beams
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_size]
        
        # Early stopping if all beams ended
        if all(beam[0][0, -1].item() == tokenizer.sep_token_id for beam in beams):
            break
            
    # Return top beam
    return beams[0][0]

# Generate answer using beam search
def generate_answer(question, model, beam_size=BEAM_SIZE):
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(f"Question: {question}", 
                      return_tensors="pt", 
                      max_length=MAX_LEN, 
                      padding="max_length", 
                      truncation=True).to(DEVICE)
    
    # Generate with beam search
    with torch.no_grad():
        output_ids = beam_search(
            model, 
            inputs["input_ids"], 
            inputs["attention_mask"], 
            beam_size=beam_size
        )
    
    # Decode output
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Ensure it starts with "Answer: "
    if not answer.startswith("Answer: "):
        answer = f"Answer: {answer}"
        
    return answer

# Main execution
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Dataset size: {len(filtered_dataset)}")
    
    # Initialize model
    model = EncoderDecoder(
        vocab_size=len(tokenizer),
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT
    )
    
    # Move model to device
    model.to(DEVICE)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    
    # Initialize scheduler
    total_steps = len(train_dataloader) * EPOCHS // ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # Train model
    print("Starting training...")
    trained_model = train_model(model, train_dataloader, val_dataloader, optimizer, scheduler)
    
    # Save final model
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, f"{CHECKPOINT_DIR}/final_model.pt")
    
    # Example inference
    print("\nTesting model with example questions:")
    example_questions = [
        "What happens when you mix baking soda and vinegar?",
        "Why do leaves change color in the fall?",
        "How does a pulley system work?"
    ]
    
    for question in example_questions:
        answer = generate_answer(question, trained_model)
        print(f"\nQ: {question}")
        print(f"A: {answer}")
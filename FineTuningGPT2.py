# ✅ Install necessary libraries
!pip install transformers datasets

import transformers
print(transformers.__version__)

# Check GPU availability and specifications
!nvidia-smi

# ✅ Imports
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import os
import logging
from datetime import datetime

# ✅ Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Create output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"./gpt2-finetuned_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# ✅ Load tokenizer and model (using standard GPT2 instead of tiny version)
model_name = "gpt2"  # Much better than tiny-gpt2
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# ✅ Ensure pad token exists
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# ✅ Load and split your training dataset
dataset = load_dataset("text", data_files={"train": "train.txt"})
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)  # 10% validation set

# ✅ Tokenize the dataset with larger context window
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128  # Increased from 64 for better context
    )

# ✅ Process datasets
tokenized_datasets = {}
for split in ["train", "test"]:
    tokenized_datasets[split] = dataset[split].map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc=f"Tokenizing {split} dataset"
    )
    logger.info(f"{split} dataset size: {len(tokenized_datasets[split])}")

# ✅ Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logger.info(f"Training on: {device}")

# ✅ Create training arguments with improved hyperparameters
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,

    # Increased training time
    num_train_epochs=10,  # Train longer (up from 3)

    # Optimized batch size
    per_device_train_batch_size=8,  # Larger batch size (up from 4)
    gradient_accumulation_steps=8,  # For effective batch size of 64

    # Learning rate schedule
    learning_rate=5e-5,  # Optimized learning rate
    warmup_steps=500,  # Warm up learning rate

    # Regularization
    weight_decay=0.01,  # Prevent overfitting

    # Evaluation and checkpointing
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,  # Save disk space

    # Mixed precision training
    fp16=torch.cuda.is_available(),  # Use half precision on GPU

    # Mitigate RAM issues
    dataloader_num_workers=4,  # Parallel data loading

    # Output all training logs
    report_to="all",
)

# ✅ Initialize Trainer with validation dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
)

# ✅ Start training
logger.info("Starting training...")
train_result = trainer.train()
training_loss = train_result.training_loss if hasattr(train_result, 'training_loss') else "N/A"
logger.info(f"Final training loss: {training_loss}")

# ✅ Save final model
trainer.save_model(os.path.join(output_dir, "final_model"))
tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
logger.info(f"Training complete. Model saved to {os.path.join(output_dir, 'final_model')}")

# ✅ Evaluate final loss on validation set manually
logger.info("Evaluating on validation set...")
model.eval()
validation_dataloader = torch.utils.data.DataLoader(
    tokenized_datasets["test"],
    batch_size=8,
    collate_fn=data_collator,
)

total_loss = 0
total_samples = 0
with torch.no_grad():
    for batch in validation_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item() * len(batch["input_ids"])
        total_samples += len(batch["input_ids"])

avg_validation_loss = total_loss / total_samples
logger.info(f"Final validation loss: {avg_validation_loss:.4f}")

# ✅ Optional: Generate sample text to test the model
if torch.cuda.is_available():  # Only do this on GPU to save time
    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate text
    output = model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        no_repeat_ngram_size=2
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info(f"Sample generated text:\n{generated_text}")

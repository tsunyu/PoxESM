import torch
import random
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForMaskedLM, DataCollatorForLanguageModeling, get_cosine_schedule_with_warmup
from Bio import SeqIO
from datasets import Dataset
import math
import numpy as np
import os
import argparse

# Add argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tuning ESM model for sequence imputation')
    parser.add_argument('--model_name', type=str, default="esm2_t33_650M_UR50D",
                        help='Name or path of the ESM model')
    parser.add_argument('--train_fasta', type=str, required=True,
                        help='Path to training FASTA file')
    parser.add_argument('--val_fasta', type=str, required=True,
                        help='Path to validation FASTA file')
    parser.add_argument('--output_dir', type=str,
                        default="./results",
                        help='Directory for output files')
    parser.add_argument('--save_model_name', type=str,
                        default="PoxESM",
                        help='Name for saved model')
    parser.add_argument('--cuda_device', type=str,
                        default="0",
                        help='CUDA device number')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    ### Load ESM-2 model and Tokenizer
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Parameters
    batch_size = 8
    warmup_epochs = 1
    num_epochs = 3
    random_seed = 16

    ### Setting random seeds for reproducibility
    # Ensure consistent results for convolution operations
    def set_random_seeds(random_seed):
        torch.manual_seed(random_seed)           # PyTorch CPU random seed
        torch.cuda.manual_seed(random_seed)      # PyTorch GPU random seed
        torch.cuda.manual_seed_all(random_seed)  # Set random seeds for all GPUs if multiple exist
        random.seed(random_seed)                 # Python random module seed
        np.random.seed(random_seed)              # Numpy random seed

        # If using CuDNN
        torch.backends.cudnn.deterministic = True  # Ensure consistent results for convolution operations
        torch.backends.cudnn.benchmark = False     # Disable benchmark to ensure reproducibility

    set_random_seeds(random_seed)

    # Read amino acid sequences
    train_sequences = []
    for record in SeqIO.parse(args.train_fasta, "fasta"):
        train_sequences.append(str(record.seq).replace('*','').replace('-',''))
    print('Number of training sequences:', len(train_sequences))

    val_sequences = []
    for record in SeqIO.parse(args.val_fasta, "fasta"):
        val_sequences.append(str(record.seq).replace('*','').replace('-',''))
    print('Number of validation sequences:', len(val_sequences))

    # Process sequences longer than 1022
    def sliding_window_truncate(lst, window_size=1022, step_size=511):
        result = []
        for item in lst:
            if len(item) > window_size:
                start = 0
                while start < len(item):
                    end = min(start + window_size, len(item))
                    result.append(item[start:end])
                    start += step_size
            else:
                result.append(item)
        return result

    train_sequences = sliding_window_truncate(train_sequences)
    print('Number of training sequences after truncation:', len(train_sequences))
    val_sequences = sliding_window_truncate(val_sequences)
    print('Number of validation sequences after truncation:', len(val_sequences))

    # Create training and test datasets
    train_dataset = Dataset.from_dict({"sequences": train_sequences})
    val_dataset = Dataset.from_dict({"sequences": val_sequences})

    # Tokenize sequences
    def tokenize_for_map(input_sequences):
        return tokenizer(input_sequences["sequences"], truncation=True, padding='max_length', max_length=1024, return_special_tokens_mask=True)

    # Tokenize training and validation datasets
    train_tokenized_datasets = train_dataset.map(tokenize_for_map, batched=True)
    val_tokenized_datasets = val_dataset.map(tokenize_for_map, batched=True)

    # Data preparation
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, return_tensors='pt', mlm=True, mlm_probability=0.15)

    # Evaluate the original model
    evaluator = Trainer(
      model=model,
      data_collator=data_collator,
      eval_dataset=val_tokenized_datasets,
      )

    eval_results_old = evaluator.evaluate()
    print(f"Original Model's Perplexity: {math.exp(eval_results_old['eval_loss']):.4f}")

    # Define optimizer
    params = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params, lr=2e-5)
    warmup_steps = math.ceil((train_tokenized_datasets.num_rows/batch_size)*warmup_epochs)
    training_steps = math.ceil((train_tokenized_datasets.num_rows/batch_size)*num_epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.001,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=10,
        evaluation_strategy="no",
        do_eval=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_datasets,
        #eval_dataset=val_tokenized_datasets,
        data_collator=data_collator,
        optimizers=[optimizer, scheduler]
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    save_dir = os.path.join(args.output_dir, args.save_model_name)
    model.save_pretrained(save_dir)

    # Evaluate the fine-tuned model
    evaluator = Trainer(
      model=model,
      data_collator=data_collator,
      eval_dataset=val_tokenized_datasets,
      )

    eval_results_new = evaluator.evaluate()

    print(f"Fine-tuned Model's Perplexity: {math.exp(eval_results_new['eval_loss']):.4f}")

if __name__ == "__main__":
    main()

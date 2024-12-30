import os
import copy
from Bio import SeqIO
from Bio.Seq import Seq
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='ESM-2 model for protein sequence imputation')
    # Required arguments
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to the input folder containing FASTA files')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to the output folder for imputed sequences')
    
    # Optional arguments
    parser.add_argument('--model_location', type=str, 
                        default='esm2_t33_650M_UR50D',
                        help='Path to the fine-tuned ESM-2 model')
    parser.add_argument('--tokenizer_location', type=str,
                        default="esm2_t33_650M_UR50D",
                        help='Path to the tokenizer')
    parser.add_argument('--cuda_device', type=str, default="0",
                        help='CUDA device number')
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    
    # Load ESM-2 model and tokenizer
    model = AutoModelForMaskedLM.from_pretrained(args.model_location)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_location)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Imputation functions
    # Sliding window parameters
    max_length = 1024
    window_size = max_length - 2  # Reserve space for special tokens
    overlap_size = 50  # Size of overlap between windows

    def fill_missing_sites(sequence, model=model, device=device):
        sequence_length = len(sequence)
        filled_sequence = list(sequence)
        if sequence_length <= max_length:
            # If sequence length is not beyond the limit, process the entire sequence
            # Replace ambiguous sites with [MASK] token
            masked_sequence = sequence.replace('X', tokenizer.mask_token).replace('x', tokenizer.mask_token)
            # Encode the sequence
            inputs = tokenizer(masked_sequence, return_tensors='pt')
            inputs = inputs.to(device)
            # Use model to predict
            with torch.no_grad():
                outputs = model(**inputs)
            predictions = outputs.logits
            # Find the position of [MASK] token
            # token indices start from 0, but 0 is <cls> special token
            mask_token_indices = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
            # Get predicted token for each MASK position
            for index in mask_token_indices:
                predicted_token_id = predictions[0, index].argmax(dim=-1)
                predicted_token = tokenizer.decode(predicted_token_id)
                # Fill in the original sequence
                filled_sequence[index.item() - 1] = predicted_token
        else:
            # If sequence length exceeds the limit, use sliding window
            for start in range(0, sequence_length, window_size - overlap_size):
                end = min(start + window_size, sequence_length)
                sub_sequence = sequence[start:end]
                sub_masked_sequence = sub_sequence.replace('X', tokenizer.mask_token).replace('x', tokenizer.mask_token)
                #sub_sequence = masked_sequence[start:end]
                # Encode the sub-sequence
                inputs = tokenizer(sub_masked_sequence, return_tensors='pt')
                inputs = inputs.to(device)         
                # Use model to predict
                with torch.no_grad():
                    outputs = model(**inputs)
                # Get predicted token ID
                predictions = outputs.logits
                # Find the position of [MASK] token
                mask_token_indices = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
                # Get predicted token for each MASK position
                for index in mask_token_indices:
                    predicted_token_id = predictions[0, index].argmax(dim=-1)
                    predicted_token = tokenizer.decode(predicted_token_id)
                    # Calculate actual position in the original sequence
                    original_position = start + index.item() - 1
                    if original_position < len(filled_sequence) and filled_sequence[original_position] == 'X':
                        filled_sequence[original_position] = predicted_token
        return ''.join(filled_sequence)

    def imputation_sequences(input_folder, output_folder):
        # Ensure output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Get all file names in the folder
        files = os.listdir(input_folder)
        
        # Filter for FASTA files (with extensions .fasta or .fa)
        fasta_files = [f for f in files if f.endswith('.fasta') or f.endswith('.fa')]
        count_imputation = 0
        count_noimputation = 0
        for fasta_file in fasta_files:
            input_file_path = os.path.join(input_folder, fasta_file)
            output_file_path = os.path.join(output_folder, fasta_file)

            print(f"Processing file: {input_file_path}")

            # Read the original FASTA file
            records = SeqIO.parse(input_file_path, "fasta")
            
            # Create new record list, filter out '*' in sequences
            new_records = []
            for record in records:
                if 'X' in str(record.seq) or 'x' in str(record.seq):
                    filtered_seq = str(record.seq).replace('*', '')
                    filled_sequence = fill_missing_sites(filtered_seq)
                    record.seq = Seq(filled_sequence + '*')
                    count_imputation += 1
                else:
                    count_noimputation += 1
                new_records.append(record)

            # Write sequences to new FASTA file
            SeqIO.write(new_records, output_file_path, "fasta")
            print(f"Output written to: {output_file_path}")

        print('Number of imputation sequences:', count_imputation)
        print('Number of no imputation sequences:', count_noimputation)

    # Call the imputation function with command line arguments
    imputation_sequences(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
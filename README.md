# ESM-2 Model Fine-tuning and Sequence Imputation

This repository contains two main scripts for protein sequence analysis using the ESM-2 language model:
1. `fineturning.py`: Fine-tunes the ESM-2 model using masked language modeling.
2. `imputation.py`: Uses the fine-tuned model to impute missing residues in protein sequences.

## Prerequisites

- Python 3.8+
- PyTorch 2.4.1+
- Transformers 4.44.1+
- BioPython
- Datasets
- NumPy

Install required packages:
```bash
pip3 install torch torchvision torchaudio
pip3 install transformers biopython datasets numpy
```


## Fine-tuning Script (fineturning.py)

This script fine-tunes the ESM-2 language model on your protein sequence dataset.

### Usage

```bash
python fineturning.py \
    --model_name "esm2_t33_650M_UR50D" \
    --train_fasta "path/to/training.fasta" \
    --val_fasta "path/to/test.fasta" \
    --output_dir "./results" \
    --save_model_name "PoxESM" \
    --cuda_device "0"
```

### Parameters

- `--model_name`: Name or path of the ESM model (default: "esm2_t33_650M_UR50D")
- `--train_fasta`: Path to training FASTA file (default: "training.fasta")
- `--val_fasta`: Path to validation FASTA file (default: "test.fasta")
- `--output_dir`: Directory for output files (default: "./results")
- `--save_model_name`: Name for saved model (default: "PoxESM")
- `--cuda_device`: CUDA device number (default: "0")

## Imputation Script (imputation.py)

This script uses the fine-tuned model to impute missing residues (marked as 'X' or 'x') in protein sequences.

### Usage

```bash
python imputation.py \
    --input_folder "path/to/input/fasta/files" \
    --output_folder "path/to/output" \
    --model_location "path/to/finetuned/model" \
    --tokenizer_location "path/to/tokenizer" \
    --cuda_device "0"
```

### Parameters

#### Required Arguments
- `--input_folder`: Path to the input folder containing FASTA files
- `--output_folder`: Path to the output folder for imputed sequences

#### Optional Arguments
- `--model_location`: Path to the fine-tuned ESM-2 model (default: "esm2_t33_650M_UR50D")
- `--tokenizer_location`: Path to the tokenizer (default: "esm2_t33_650M_UR50D")
- `--cuda_device`: CUDA device number (default: "0")

### Features

- Handles sequences of any length using a sliding window approach
- Processes multiple FASTA files in batch
- Maintains sequence format and headers
- Preserves terminal stop codons ('*')
- Reports statistics on processed sequences

## Notes

- For sequences longer than 1022 residues, a sliding window approach is used with 50-residue overlap
- The model uses masked language modeling with a 15% masking probability during fine-tuning
- GPU acceleration is recommended for better performance

## Citation 

If you use this code, please cite the following paper:

```
Tracking Clade Ib Mpox Virus Evolution Through a Protein Language Model, Junyu Luo, Xiyang Cai, Yixue Li, 2024, In Preparation
```




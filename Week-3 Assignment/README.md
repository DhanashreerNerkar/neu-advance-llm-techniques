# Implementation of "Self-Alignment with Instruction Backtranslation"

This project is an implementation of the ICLR 2024 paper, **"Self-Alignment with Instruction Backtranslation"** (https://arxiv.org/pdf/2308.06259.pdf).

The goal is to create a high-quality, instruction-following language model using a novel, 4-step pipeline. This process uses a base model to generate and curate its own training data, effectively teaching itself to be a better assistant.

This implementation was built entirely in Google Colab using `transformers`, 4-bit quantization (`bitsandbytes`), and LoRA (`peft`) for all fine-tuning steps.

## The 4-Step Pipeline

The project is broken into four distinct, sequential stages:

### Step 1: Fine-tuning the Backward Model ($M_{yx}$)
A base `meta-llama/Llama-2-7b-hf` model was fine-tuned on the `timdettmers/openassistant-guanaco` dataset. Instead of the standard `(instruction, output)` format, it was trained on `(output, instruction)` pairs. This creates a "backward model" ($M_{yx}$) that learns to predict a likely *instruction* given a piece of *text*.

### Step 2: Self-Augmentation
The backward model from Step 1 was used to generate 150 new instructions. It was fed 150 high-quality responses (completions) from the `databricks/databricks-dolly-15k` dataset, and it predicted a probable instruction for each one. This created a new, synthetic dataset of 150 `(generated_instruction, original_response)` pairs.

### Step 3: Self-Curation
The 150 synthetic pairs were then "judged" for quality. The `meta-llama/Llama-2-7b-chat-hf` model was used as a rater, prompted with few-shot examples and criteria from the paper (Table 19). It assigned a score from 1-5 to each pair. All pairs scoring 4 or 5 were kept, filtering the set down to **93 high-quality examples**.

### Step 4: Final Model Fine-tuning
A new, base `meta-llama/Llama-2-7b-hf` model was fine-tuned with LoRA on the 93 curated examples from Step 3. This final model is the result of the self-alignment pipeline, trained only on a small, high-quality dataset that the model itself helped generate and filter.

---

## Final Assets on Hugging Face Hub

This project produced three key assets, all available on the Hugging Face Hub:

1.  **Backward Model (Step 1):**
    * **URL:** `https://huggingface.co/dnerkar/llama-2-7b-backward-lora`

2.  **Curated Dataset (Step 3):**
    * **URL:** `https://huggingface.co/datasets/dnerkar/self-aligned-curated-dataset`

3.  **Final Aligned Model (Step 4):**
    * **URL:** `https://huggingface.co/dnerkar/llama-2-7b-self-aligned-model`

---

## Technology Stack

* **Platform:** Google Colab (T4 GPU)
* **Models:**
    * `meta-llama/Llama-2-7b-hf` (Base model for fine-tuning)
    * `meta-llama/Llama-2-7b-chat-hf` (Judge model for curation)
* **Datasets:**
    * `timdettmers/openassistant-guanaco` (Seed data for backward model)
    * `databricks/databricks-dolly-15k` (Source of unlabeled completions)
* **Key Libraries:**
    * `transformers`
    * `peft` (LoRA)
    * `bitsandbytes` (4-bit QLoRA)
    * `datasets`
    * `accelerate`
    * `torch`

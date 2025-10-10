import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import re
import datetime
import json
import warnings

import threading
import time
import subprocess
import sys
import os

class OdinAssistant:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """
        Initialize Odin AI Assistant with proper attention mask handling
        """
        self.model_name = model_name
        # If CUDA not available, fallback to a smaller model to avoid long CPU training
        if not torch.cuda.is_available():
            print("No CUDA detected — falling back to smaller model for CPU training (distilgpt2)")
            self.model_name = "distilgpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Properly set pad token to avoid attention mask issues
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"Odin initialized on {self.device}")
        print(f"Pad token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")

        # For continual learning
        self.user_examples = []
        self.training_lock = threading.Lock()
        # Hardware-aware defaults (will be overridden by detection)
        self.hardware = {
            "has_cuda": torch.cuda.is_available(),
            "cuda_device": 0 if torch.cuda.is_available() else None,
            "gpu_total_mem_mb": None,
            "system_ram_gb": None,
        }
        # Try to detect GPU memory
        try:
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(self.device).total_memory
                self.hardware["gpu_total_mem_mb"] = int(gpu_mem / 1024**2)
        except Exception:
            self.hardware["gpu_total_mem_mb"] = None
        # Try to detect system RAM
        try:
            import psutil
            self.hardware["system_ram_gb"] = int(psutil.virtual_memory().total / (1024**3))
        except Exception:
            self.hardware["system_ram_gb"] = None
        print(f"Hardware: {self.hardware}")
        # Enable gradient checkpointing if available to save memory
        try:
            if torch.cuda.is_available() and hasattr(self.model, "gradient_checkpointing_enable"):
                # gradient checkpointing requires use_cache=False
                try:
                    self.model.config.use_cache = False
                except Exception:
                    pass
                self.model.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled (use_cache set to False)")
            else:
                print("Gradient checkpointing not enabled (requires CUDA)")
        except Exception:
            pass

    @staticmethod
    def continual_fine_tune_process(interval=60, output_dir="./odin-finetuned-user"):
        """Standalone process: periodically fine-tune on user examples from file"""
        import json
        import time
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
        import torch
            
        try:
            from datasets import Dataset as HFDataset
        except Exception:
            HFDataset = None
        # Choose model based on availability of CUDA
        model_name = "microsoft/DialoGPT-medium"
        if not torch.cuda.is_available():
            model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # Ensure pad token exists for batch tokenization
        if tokenizer.pad_token is None:
            try:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
                print("[Continual Trainer] Added PAD token and resized model embeddings")
            except Exception:
                # Fallback: set pad_token to eos_token if resizing fails
                try:
                    tokenizer.pad_token = tokenizer.eos_token
                except Exception:
                    pass
        user_data_file = "user_data.jsonl"
        print("[Continual Trainer] Started in new terminal.")
        while True:
            time.sleep(interval)
            if not os.path.exists(user_data_file):
                continue
            with open(user_data_file, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            if not lines:
                continue
            texts = [f"User: {ex['user']}\nAssistant: {ex['assistant']}" for ex in lines]
            dataset = [{"text": t} for t in texts]
            print(f"[Continual Trainer] Fine-tuning on {len(dataset)} user examples...")
            # Prefer Hugging Face Dataset so Trainer and DataCollator can work naturally
            if HFDataset is not None:
                hf_dataset = HFDataset.from_list(dataset)
                # tokenize on the fly with map; keep in memory to avoid writing Arrow mmap files on Windows
                def tokenize_fn(examples):
                    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
                hf_dataset = hf_dataset.map(tokenize_fn, batched=True, keep_in_memory=True, load_from_cache_file=False)
                # Keep only needed columns
                if "text" in hf_dataset.column_names:
                    hf_dataset = hf_dataset.remove_columns([c for c in hf_dataset.column_names if c not in ("input_ids", "attention_mask")])
                torch_dataset = hf_dataset
            else:
                # Fallback: small in-memory tensors
                processed = [tokenizer(ex["text"], truncation=True, padding="max_length", max_length=512) for ex in dataset]
                input_id_list = [torch.tensor(d["input_ids"]) for d in processed]
                attention_list = [torch.tensor(d["attention_mask"]) for d in processed]
                input_ids = torch.stack(input_id_list)
                attention_mask = torch.stack(attention_list)
                torch_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask)
            # Choose training args conservatively for continual updates
            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                logging_steps=1,
                save_steps=10,
                save_total_limit=1,
                fp16=False,
                dataloader_pin_memory=False
            )
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=torch_dataset,
                tokenizer=tokenizer
            )
            trainer.train()
            # Save with retries to avoid Windows file-mapped collisions
            def save_with_retries(trainer_obj, tokenizer_obj, base_dir, max_attempts=3):
                for attempt in range(1, max_attempts+1):
                    try:
                        trainer_obj.save_model(base_dir)
                        tokenizer_obj.save_pretrained(base_dir)
                        return base_dir
                    except Exception as e:
                        print(f"[Continual Trainer] Save attempt {attempt} failed: {e}")
                        # fallback to timestamped directory
                        base_dir = f"{base_dir.rstrip('/')}-{int(datetime.datetime.now().timestamp())}"
                        time.sleep(1)
                # final attempt: try once more and propagate exception
                trainer_obj.save_model(base_dir)
                tokenizer_obj.save_pretrained(base_dir)
                return base_dir

            try:
                outdir = save_with_retries(trainer, tokenizer, output_dir)
                print(f"[Continual Trainer] Model updated with user data. Saved to {outdir}")
            except Exception as e:
                print(f"[Continual Trainer] Failed to save model after retries: {e}")

            # Clear file after training
            open(user_data_file, "w").close()

    def add_user_example(self, user, assistant):
        # Append to file for cross-process sharing
        import json
        with open("user_data.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"user": user, "assistant": assistant}) + "\n")

    def continual_fine_tune(self, interval=60, output_dir="./odin-finetuned-user"):
        """Background thread: periodically fine-tune on user examples"""
        while True:
            time.sleep(interval)
            with self.training_lock:
                if not self.user_examples:
                    continue
                # Prepare a mini dataset from user_examples
                texts = [f"User: {ex['user']}\nAssistant: {ex['assistant']}" for ex in self.user_examples]
                dataset = [{"text": t} for t in texts]
                print(f"[Continual Trainer] Fine-tuning on {len(dataset)} user examples...")
                def tokenize_function(examples):
                    return self.tokenizer(
                        examples["text"],
                        truncation=True,
                        padding="max_length",
                        max_length=512
                    )
                # Simulate HuggingFace dataset map
                class DummyDataset(torch.utils.data.Dataset):
                    def __init__(self, data):
                        self.data = data
                    def __len__(self):
                        return len(self.data)
                    def __getitem__(self, idx):
                        return self.data[idx]
                processed = list(map(lambda ex: self.tokenizer(
                    ex["text"], truncation=True, padding="max_length", max_length=512
                ), dataset))
                # Convert to torch tensors
                input_ids = torch.stack([torch.tensor(d["input_ids"]) for d in processed])
                attention_mask = torch.stack([torch.tensor(d["attention_mask"]) for d in processed])
                torch_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask)
                # TrainingArguments
                training_args = TrainingArguments(
                    output_dir=output_dir,
                    overwrite_output_dir=True,
                    num_train_epochs=1,
                    per_device_train_batch_size=1,
                    logging_steps=1,
                    save_steps=10,
                    save_total_limit=1,
                    fp16=False,
                    dataloader_pin_memory=False
                )
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False
                )
                trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=torch_dataset,
                    tokenizer=self.tokenizer
                )
                trainer.train()
                trainer.save_model()
                self.tokenizer.save_pretrained(output_dir)
                print(f"[Continual Trainer] Model updated with user data.")
                self.user_examples.clear()

    def generate_response(self, prompt, max_length=150):
        """Generate response to user input with proper attention masking"""
        # Encode with attention mask
        inputs = self.tokenizer(
            prompt, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Use safer decoding settings to reduce repetition
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.8,
                top_p=0.92,
                top_k=50,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # early_stopping is not valid for some generation configs; omit it
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        # Basic post-processing to collapse repeated characters and repeated words
        # Collapse long repeated characters (e.g., "aaaaa" -> "a")
        response = re.sub(r"(.)\1{3,}", r"\1", response)
        # Collapse repeated word sequences longer than 2 repeats: "hi hi hi" -> "hi"
        response = re.sub(r"\b(\w+)(?:\s+\1){2,}\b", r"\1", response, flags=re.IGNORECASE)
        # Trim excessive whitespace
        response = re.sub(r"\s{2,}", " ", response).strip()

        return response

    def prepare_dataset(self, dataset_name="nikhilkeetha/personalized-assistant", split="train"):
        """Load and preprocess dataset for training"""
        print(f"Loading dataset {dataset_name}...")
        try:
            # Use SQuAD dataset which is reliable and well-structured
            dataset = load_dataset("squad", split=split)
            
            # Take a very small subset for initial testing (fast smoke test)
            dataset = dataset.shuffle(seed=42).select(range(20))  # Start with 20 examples
            print(f"Selected {len(dataset)} examples for training")
            
            def process_conversations(example):
                # Create conversational format from question and answer
                conversation = f"User: {example['question']}\nAssistant: {example['answers']['text'][0]}"
                return {"text": conversation.strip()}
            
            print("Processing dataset...")
            # Process dataset with progress bar
            processed_dataset = dataset.map(
                process_conversations,
                desc="Converting to conversation format"
            )
            print("Dataset processing completed")
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            # Try a different dataset as fallback
            try:
                dataset = load_dataset("nikhilkeetha/personalized-assistant", split=split)
                def process_squad(example):
                    conversation = f"User: {example['question']}\nAssistant: {example['answers']['text'][0]}"
                    return {"text": conversation.strip()}
                processed_dataset = dataset.map(process_squad)
            except Exception as e2:
                print(f"Error loading fallback dataset: {str(e2)}")
                raise e
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512
                # Removed return_tensors="pt" - this was causing the issue
            )
        
        tokenized_dataset = processed_dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=processed_dataset.column_names
        )
        return tokenized_dataset

    def fine_tune(self, dataset_name="daily_dialog", output_dir="./odin-finetuned"):
        """Fine-tune Odin on conversational dataset"""
        print("Preparing dataset...")
        train_dataset = self.prepare_dataset(dataset_name)
        print(f"Dataset size: {len(train_dataset)} examples")
        
        # Calculate optimal batch size based on available memory
        batch_size = 1 if torch.cuda.is_available() else 1
        
        print("Configuring training arguments based on hardware...")
        # Decide fp16 and batch sizing heuristics (RTX 2060 ~6GB VRAM)
        gpu_mem = self.hardware.get("gpu_total_mem_mb") or 0
        system_ram = self.hardware.get("system_ram_gb") or 0
        use_fp16 = False
        per_device_batch = 1
        gradient_accumulation = 1
        # If we have >= 8GB VRAM, allow fp16 and slightly larger batch
        if gpu_mem and gpu_mem >= 8192:
            use_fp16 = True
            per_device_batch = 2
            gradient_accumulation = 2
        # If around 6GB (RTX 2060), be conservative
        elif gpu_mem and 5000 <= gpu_mem < 8192:
            use_fp16 = False
            per_device_batch = 1
            gradient_accumulation = 2
        else:
            # No GPU or unknown -> CPU training, keep minimal
            use_fp16 = False
            per_device_batch = 1
            gradient_accumulation = 4

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=per_device_batch,
            gradient_accumulation_steps=gradient_accumulation,
            save_steps=10,
            save_total_limit=2,
            logging_steps=1,
            fp16=use_fp16,
            dataloader_pin_memory=bool(torch.cuda.is_available()),
            logging_dir=f"{output_dir}/logs"
        )
        print(f"Training config: fp16={use_fp16}, batch={per_device_batch}, accum={gradient_accumulation}")
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer
        )
        
        print("Starting fine-tuning...")
        trainer.train()
        # Save with retries (Windows may lock files due to mmap)
        def save_with_retries_main(trainer_obj, tokenizer_obj, base_dir, max_attempts=3):
            for attempt in range(1, max_attempts+1):
                try:
                    trainer_obj.save_model(base_dir)
                    tokenizer_obj.save_pretrained(base_dir)
                    return base_dir
                except Exception as e:
                    print(f"Save attempt {attempt} failed: {e}")
                    base_dir = f"{base_dir.rstrip('/')}-{int(datetime.datetime.now().timestamp())}"
                    time.sleep(1)
            trainer_obj.save_model(base_dir)
            tokenizer_obj.save_pretrained(base_dir)
            return base_dir

        outdir = save_with_retries_main(trainer, self.tokenizer, output_dir)
        print(f"Model saved to {outdir}")

def main():

    # Suppress some warnings for cleaner output
    warnings.filterwarnings("ignore", message=".*padding_side.*")

    # Read dataset names from datasets.txt
    dataset_file = "datasets.txt"
    try:
        with open(dataset_file, "r") as f:
            dataset_names = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except Exception as e:
        print(f"Could not read {dataset_file}: {e}")
        dataset_names = ["squad"]

    # Initialize Odin
    odin = OdinAssistant()

    # Fine-tune on each dataset (initial training)
    for dataset_name in dataset_names:
        print(f"\n--- Fine-tuning on dataset: {dataset_name} ---")
        try:
            odin.fine_tune(dataset_name=dataset_name)
        except Exception as e:
            print(f"Failed to fine-tune on {dataset_name}: {e}")

    # Start continual training in a new terminal
    python_exe = sys.executable
    script_path = os.path.abspath(__file__)
    subprocess.Popen([
        "start", "cmd", "/k", f"{python_exe} {script_path} --continual-train"
    ], shell=True)

    print("\nOdin AI Assistant is ready! Type 'exit' to quit.")
    conversation_history = []

    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Odin: Farewell! Until next time.")
                break

            # Build context with conversation history
            context = "\n".join(conversation_history[-4:])  # Last 2 exchanges
            full_prompt = f"{context}\nUser: {user_input}\nAssistant:"

            response = odin.generate_response(full_prompt)
            print(f"Odin: {response}")

            # Update conversation history
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Assistant: {response}")

            # Add user interaction for continual training
            odin.add_user_example(user_input, response)
    except KeyboardInterrupt:
        print("\nInterrupted — exiting gracefully.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--continual-train":
        OdinAssistant.continual_fine_tune_process()
    else:
        main()

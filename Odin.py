import os
# Set transformers verbosity env before importing transformers to suppress
# certain model-internal docstring/argument warnings that are noisy at startup.
# Prefer environment-level suppression to editing site-packages.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Note: imports for heavy libraries (torch, transformers, datasets) are performed
# later inside a guarded block where stdout/stderr are temporarily captured to
# avoid printing advisory messages from transformers internals during import.
import re
import platform
import datetime
import io
import sys
import warnings
import threading
# A small stream wrapper that filters out noisy advisory lines emitted during
# transformers import/model loading. We keep it minimal: buffer writes until a
# newline and only forward lines that don't match any suppressed pattern.
import re as _re


class _FilteredStream:
    def __init__(self, underlying, block_patterns=None):
        self._underlying = underlying
        # compile patterns for faster checks
        self._patterns = [(_re.compile(p) if isinstance(p, str) else p) for p in (block_patterns or [])]
        self._buffer = ""

    def write(self, s):
        try:
            self._buffer += s
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                # decide whether to forward this line
                if any(p.search(line) for p in self._patterns):
                    # drop the line
                    continue
                try:
                    self._underlying.write(line + "\n")
                except Exception:
                    pass
        except Exception:
            # Fallback: try to write raw
            try:
                self._underlying.write(s)
            except Exception:
                pass

    def flush(self):
        try:
            if self._buffer:
                # flush remaining buffer
                line = self._buffer
                self._buffer = ""
                if not any(p.search(line) for p in self._patterns):
                    self._underlying.write(line)
            self._underlying.flush()
        except Exception:
            pass

# Temporarily capture stdout/stderr during imports to suppress noisy
# advisory messages emitted by some transformers model modules at import time
# (for example: the 'high_res_pixel_values' advisory). This is non-invasive and
# avoids editing site-packages. We restore streams immediately after imports.
_saved_stdout, _saved_stderr = sys.stdout, sys.stderr
try:
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,  # Using seq2seq for T5
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,  # T5-specific collator
        DataCollatorForLanguageModeling  # For LM training
    )
    # Suppress overly-verbose or undocumented-argument warnings from transformers internals
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    try:
        from datasets import load_dataset
    except Exception:
        load_dataset = None
finally:
    # restore streams so normal prints show up again
    sys.stdout, sys.stderr = _saved_stdout, _saved_stderr
    # If datasets was not available, warn once (outside suppressed block)
    if load_dataset is None:
        print("[Startup] Warning: 'datasets' package not available. Dataset loading and fine-tuning will use fallbacks.")
    try:
        import multiprocess as _mp
        _patched = False
        for mod_attr in (getattr(_mp, 'pickle', None), getattr(_mp, 'reduction', None), _mp):
            if mod_attr is None:
                continue
            # search attributes for a Pickler-like class
            for name in dir(mod_attr):
                try:
                    obj = getattr(mod_attr, name)
                except Exception:
                    continue
                if not isinstance(obj, type):
                    continue
                if hasattr(obj, '_batch_setitems'):
                    orig = getattr(obj, '_batch_setitems')
                    # only patch if callable
                    if callable(orig):
                        def _make_wrapper(original):
                            def _wrapper(self, *args, **kwargs):
                                try:
                                    return original(self, *args, **kwargs)
                                except TypeError:
                                    # Try to adapt: if original expects fewer args, call with first two positional args
                                    try:
                                        if len(args) >= 2:
                                            return original(self, args[0], args[1])
                                        elif len(args) == 1:
                                            return original(self, args[0])
                                    except Exception:
                                        # As a last resort, call original with only self
                                        return original(self)
                            return _wrapper
                        try:
                            setattr(obj, '_batch_setitems', _make_wrapper(orig))
                            print("Patched multiprocess Pickler._batch_setitems for flexible signature handling")
                            _patched = True
                            break
                        except Exception:
                            # ignore and continue searching
                            pass
            if _patched:
                break
    except Exception:
        # If multiprocess not available or patch fails, continue without error.
        pass

# Extra sweep: scan already-imported modules for Pickler classes with _batch_setitems
try:
    import sys as _sys
    for mod in list(_sys.modules.values()):
        try:
            for name in dir(mod):
                try:
                    obj = getattr(mod, name)
                except Exception:
                    continue
                if not isinstance(obj, type):
                    continue
                if hasattr(obj, '_batch_setitems'):
                    orig = getattr(obj, '_batch_setitems')
                    if callable(orig):
                        def _make_wrapper2(original):
                            def _wrapper(self, *args, **kwargs):
                                try:
                                    return original(self, *args, **kwargs)
                                except TypeError:
                                    try:
                                        if len(args) >= 2:
                                            return original(self, args[0], args[1])
                                        elif len(args) == 1:
                                            return original(self, args[0])
                                    except Exception:
                                        return original(self)
                            return _wrapper
                        try:
                            setattr(obj, '_batch_setitems', _make_wrapper2(orig))
                            print(f"Patched Pickler._batch_setitems on {obj} from module {getattr(mod,'__name__', repr(mod))}")
                        except Exception:
                            pass
        except Exception:
            continue
except Exception:
    pass

class OdinAssistant:
    def __init__(self, model_name="google/flan-t5-large", finetuned_dir=None, force_cpu=False):
        """
        Initialize Odin AI Assistant with FLAN-T5 for better knowledge-based responses
        
        Args:
            model_name: Base model name to use
            finetuned_dir: Directory containing finetuned model (optional)
            force_cpu: Force CPU usage to avoid CUDA errors (optional)
        """
        # Keep the original base model name so we can fall back to it if a finetuned model
        # produces poor/garbled outputs.
        self.base_model_name = model_name
        self.model_name = model_name
        self.finetuned_dir = finetuned_dir

        # If a finetuned directory is provided and exists, prefer loading from it
        loaded_from_finetuned = False
        if finetuned_dir:
            try:
                if os.path.isdir(finetuned_dir):
                    print(f"Loading tokenizer/model from finetuned dir: {finetuned_dir}")
                    # Suppress noisy advisory prints during model/tokenizer load
                    _saved_out, _saved_err = sys.stdout, sys.stderr
                    try:
                        sys.stdout = io.StringIO()
                        sys.stderr = io.StringIO()
                        self.tokenizer = AutoTokenizer.from_pretrained(finetuned_dir)
                        self.model = AutoModelForSeq2SeqLM.from_pretrained(finetuned_dir)
                    finally:
                        sys.stdout, sys.stderr = _saved_out, _saved_err
                    self.model_name = finetuned_dir
                    loaded_from_finetuned = True
                else:
                    print(f"Finetuned dir '{finetuned_dir}' does not exist; falling back to base model")
            except Exception as e:
                print(f"Failed to load finetuned model from {finetuned_dir}: {e}\nFalling back to base model.")

        if not loaded_from_finetuned:
            # If CUDA not available, fallback to base-small for CPU
            if not torch.cuda.is_available():
                print("No CUDA detected — falling back to smaller model (google/flan-t5-base)")
                self.model_name = "google/flan-t5-base"
            # Suppress noisy advisory prints during model/tokenizer load
            _saved_out2, _saved_err2 = sys.stdout, sys.stderr
            try:
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            finally:
                sys.stdout, sys.stderr = _saved_out2, _saved_err2
            
        # T5 models come with padding token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Properly set pad token to avoid attention mask issues
        # Properly handle tokenizer padding and embedding resizing for tensor core optimization
        if self.tokenizer.pad_token is None:
            # Add pad token while ensuring embedding dimension is optimized for tensor cores
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # Calculate next multiple of 8 for tensor core optimization
            vocab_size = len(self.tokenizer)
            pad_to = 8
            optimized_size = ((vocab_size + pad_to - 1) // pad_to) * pad_to
            
            print(f"Resizing embedding layer from {self.model.get_input_embeddings().weight.shape[0]} to {optimized_size} for tensor core optimization")
            self.model.resize_token_embeddings(optimized_size, pad_to_multiple_of=pad_to)
            
        # Move model to GPU with optimized settings if available
        if force_cpu:
            print("Forcing CPU usage as requested")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda" and not force_cpu:
            try:
                # Clear CUDA cache first to avoid memory issues
                torch.cuda.empty_cache()
                
                # Move model to GPU with error handling
                self.model = self.model.to(self.device)
                
                # Skip half precision and memory format optimization to avoid CUDA errors
                print("Model moved to CUDA successfully (using float32 for stability)")
                
            except Exception as e:
                print(f"Warning: Could not move model to CUDA: {e}")
                print("Falling back to CPU for stability")
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
        else:
            self.model = self.model.to(self.device)
            if force_cpu:
                print("Using CPU mode for maximum stability")        # Explicit CUDA debug info so users can quickly verify GPU usage
        try:
            cuda_avail = torch.cuda.is_available()
            cuda_version = getattr(torch.version, 'cuda', None)
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if cuda_avail and device_count > 0 else None
            device_props = None
            if cuda_avail and device_count > 0:
                try:
                    props = torch.cuda.get_device_properties(0)
                    device_props = {
                        'name': props.name,
                        'total_memory_mb': int(props.total_memory / 1024**2),
                        'multi_processor_count': getattr(props, 'multi_processor_count', None)
                    }
                except Exception:
                    device_props = None
        except Exception:
            cuda_avail = False
            cuda_version = None
            device_count = 0
            device_name = None
            device_props = None

        print(f"Odin initialized on {self.device}")
        print(f"Pad token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
        print(f"CUDA available: {cuda_avail}, torch.cuda.version: {cuda_version}, device_count: {device_count}")
        if device_name:
            print(f"CUDA device[0]: {device_name}, properties: {device_props}")

        # For continual learning
        self.user_examples = []
        self.training_lock = threading.Lock()
        
        # Personality and conversation state for more human-like interactions
        self.conversation_mood = "friendly"  # Can be: friendly, enthusiastic, thoughtful, etc.
        self.user_interests = set()  # Track what the user seems interested in
        self.conversation_depth = 0  # Track how deep into a topic we are
        self.last_topic = None  # Remember what we were just talking about
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
        # Enable gradient checkpointing with better error handling
        try:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                # Set use_cache=False for gradient checkpointing
                try:
                    self.model.config.use_cache = False
                except Exception:
                    pass
                
                # Enable gradient checkpointing with error handling
                try:
                    self.model.gradient_checkpointing_enable()
                    print("Gradient checkpointing enabled safely")
                except Exception as e:
                    print(f"Warning: Could not enable gradient checkpointing: {e}")
                    # Re-enable cache if checkpointing fails
                    try:
                        self.model.config.use_cache = True
                    except Exception:
                        pass
            else:
                print("Gradient checkpointing not available for this model")
        except Exception as e:
            print(f"Warning: Gradient checkpointing setup failed: {e}")

    @staticmethod
    def continual_fine_tune_process(interval=60, output_dir="./odin-finetuned-user"):
        """Standalone process: periodically fine-tune on user examples from file"""
        import json
        import time
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
        import torch
            
        try:
            from datasets import Dataset as HFDataset
        except Exception:
            HFDataset = None
        # Choose model based on availability of CUDA
        model_name = "google/flan-t5-large"
        if not torch.cuda.is_available():
            model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
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
                train_dataset=torch_dataset
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
        
        # Analyze user interests for more personalized responses
        self._analyze_user_interests(user)
    
    def _analyze_user_interests(self, user_input):
        """Analyze user input to understand their interests and adapt conversation style"""
        user_lower = user_input.lower()
        
        # Track health/wellness interests
        health_keywords = ['herb', 'supplement', 'nutrition', 'diet', 'wellness', 'health', 'natural', 'remedy', 'healing', 'medicine']
        if any(keyword in user_lower for keyword in health_keywords):
            self.user_interests.add('natural_health')
        
        # Track specific areas of interest
        if any(word in user_lower for word in ['stress', 'anxiety', 'calm', 'relax']):
            self.user_interests.add('stress_management')
        if any(word in user_lower for word in ['energy', 'tired', 'fatigue', 'sleep']):
            self.user_interests.add('energy_sleep')
        if any(word in user_lower for word in ['digestion', 'stomach', 'gut', 'digestive']):
            self.user_interests.add('digestive_health')
        if any(word in user_lower for word in ['immune', 'cold', 'flu', 'infection']):
            self.user_interests.add('immune_support')
        
        # Adjust conversation mood based on user's tone
        if any(word in user_lower for word in ['excited', 'amazing', 'love', 'fantastic', '!']):
            self.conversation_mood = "enthusiastic"
        elif any(word in user_lower for word in ['worried', 'concerned', 'problem', 'help', 'struggling']):
            self.conversation_mood = "supportive"
        elif any(word in user_lower for word in ['research', 'study', 'evidence', 'scientific']):
            self.conversation_mood = "thoughtful"
        else:
            self.conversation_mood = "friendly"

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

    def generate_response(self, prompt, max_length=512, response_format="json", ensure_relevant=False, max_attempts=3):
        """
        Generate a response using FLAN-T5's instruction-following capabilities.
        Formats prompt as a clear instruction and handles both JSON and plain text outputs.
        """
        # Handle identity questions with a more natural, conversational response
        # Only trigger for very specific identity questions
        if re.search(r"^(who\s+are\s+you|what\s+are\s+you|introduce\s+yourself)[\?\.]?\s*$", prompt.strip(), flags=re.I):
            identity_responses = [
                "Hey there! I'm Odin, your friendly AI assistant. I love chatting about health, wellness, and natural approaches to feeling your best. What's on your mind today?",
                "Hi! I'm Odin - think of me as your knowledgeable friend who's really into natural health and wellness. I'm here to help with whatever questions you have!",
                "Hello! I'm Odin, and I'm passionate about natural medicine and holistic health. I enjoy having real conversations and helping people learn. What would you like to explore together?",
                "Hey! I'm Odin - I'm an AI, but I like to think I'm a pretty good conversationalist. I specialize in natural health and wellness topics. How can I help you today?"
            ]
            import random
            canned = random.choice(identity_responses)
            if response_format == "json":
                resp_obj = {
                    "message": canned,
                    "metadata": {"length": len(canned), "timestamp": datetime.datetime.utcnow().isoformat() + "Z", "model": self.model_name, "device": str(self.device)}
                }
                try:
                    return json.dumps(resp_obj, ensure_ascii=False)
                except Exception:
                    return json.dumps(resp_obj)
            return canned

        # Adapt response style based on conversation context and user interests
        mood_modifiers = {
            "enthusiastic": "Respond with enthusiasm and energy, using exclamation points and positive language.",
            "supportive": "Respond with empathy and care, offering gentle guidance and reassurance.",
            "thoughtful": "Respond with depth and scientific backing, including research references when relevant.",
            "friendly": "Respond in a warm, conversational tone like talking to a good friend."
        }
        
        interest_context = ""
        if self.user_interests:
            interests_list = list(self.user_interests)
            if len(interests_list) == 1:
                interest_context = f"The user has shown interest in {interests_list[0].replace('_', ' ')}. "
            else:
                interest_context = f"The user has shown interest in {', '.join(interests_list).replace('_', ' ')}. "

        # More conversational and natural instruction template
        # Very simple, direct instruction format
        if self.conversation_mood == "enthusiastic":
            personality_note = "I'm excited to share what I know! "
        elif self.conversation_mood == "supportive":
            personality_note = "I'm here to help and support you. "
        elif self.conversation_mood == "thoughtful":
            personality_note = "Let me share some thoughtful insights. "
        else:
            personality_note = ""
            
        instruction_template = """{personality_note}Question: {query}

Answer:"""

        # Simpler, more direct examples that show the conversational style
        few_shot_examples = [
            ("Tell me about natural medicine", 
             "Oh, I love this topic! Natural medicine is all about working with your body's incredible ability to heal itself. Think herbs, nutrition, lifestyle changes - basically nature's toolkit for wellness. What's really cool is how it looks at the whole person, not just symptoms. It's been around forever, and now science is catching up with what traditional healers always knew. What got you interested in natural approaches?"),
            
            ("What herbs help with stress?",
             "Great question! Some of my favorites for stress are ashwagandha - it's like a chill pill from nature - and chamomile, which is super gentle and calming. Lemon balm is wonderful too, and passionflower can really help quiet a busy mind. The key is finding what works for your body. Are you dealing with stress right now, or just curious about natural options?")
        ]
        # Use a much simpler approach that works better with T5
        # T5 works best with direct, simple prompts
        full_prompt = f"Answer this question about natural health in a friendly, conversational way: {prompt}"

        # Prepare input for T5 with proper padding and attention mask
        inputs = self.tokenizer(
            full_prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Conservative T5 generation settings for stable output
        generation_config = {
            "max_length": min(max_length, 200),  # Cap max length
            "min_length": 30,  # Reasonable minimum
            "do_sample": True,
            "top_p": 0.9,      # Higher for more natural language
            "top_k": 50,       # Broader vocabulary
            "temperature": 0.8, # Moderate temperature
            "num_beams": 3,    # Use beam search for more coherent output
            "length_penalty": 1.0,
            "repetition_penalty": 1.1,  # Light repetition penalty
            "no_repeat_ngram_size": 2,  # Prevent short repetitions
            "early_stopping": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }

        def _single_generate(inputs_local, min_new_tokens=60):
            # Use safer generation settings to avoid CUDA errors
            try:
                # Clear CUDA cache before generation to prevent memory issues
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Use no_grad context for all generation to avoid gradient issues
                with torch.no_grad():
                    generation_config = {
                        "max_new_tokens": min(max_length, 150),
                        "min_new_tokens": max(20, min_new_tokens),
                        "do_sample": True,
                        "top_p": 0.9,
                        "top_k": 50,
                        "temperature": 0.8,
                        "num_beams": 3,     # Use beam search for coherence
                        "length_penalty": 1.0,
                        "early_stopping": True,
                        "no_repeat_ngram_size": 2,
                        "repetition_penalty": 1.1,
                        "num_return_sequences": 1,
                        "pad_token_id": self.tokenizer.pad_token_id,
                        "eos_token_id": self.tokenizer.eos_token_id,
                        "use_cache": getattr(self.model.config, 'use_cache', True)
                    }

                    # Simple generation without CUDA optimizations to avoid errors
                    outputs = self.model.generate(**inputs_local, **generation_config)
                    
            except Exception as e:
                print(f"[Generation] Exception during generation on device {self.device}: {e}")
                # If CUDA error, try moving to CPU temporarily
                if "cuda" in str(e).lower() or "assert" in str(e).lower():
                    print("[Generation] CUDA error detected, attempting CPU fallback...")
                    try:
                        # Move inputs to CPU
                        cpu_inputs = {k: v.cpu() for k, v in inputs_local.items()}
                        # Temporarily move model to CPU
                        original_device = self.model.device
                        self.model = self.model.cpu()
                        
                        with torch.no_grad():
                            cpu_config = generation_config.copy()
                            cpu_config["use_cache"] = True  # Enable cache for CPU
                            outputs = self.model.generate(**cpu_inputs, **cpu_config)
                        
                        # Move model back to original device
                        self.model = self.model.to(original_device)
                        # Move outputs back to original device
                        outputs = outputs.to(original_device)
                        
                    except Exception as cpu_e:
                        print(f"[Generation] CPU fallback also failed: {cpu_e}")
                        return None
                else:
                    return None
            # outputs contains the full sequence (prompt + generated). Compute the actual prompt length
            try:
                # Use attention_mask to count non-padded tokens when available
                if 'attention_mask' in inputs_local:
                    input_len = int(inputs_local['attention_mask'].sum().item())
                else:
                    input_len = inputs_local['input_ids'].ne(self.tokenizer.pad_token_id).sum().item()
            except Exception:
                input_len = inputs_local['input_ids'].shape[-1]

            # Ensure we don't slice beyond bounds
            total_len = outputs.shape[-1]
            if input_len < total_len:
                gen_ids = outputs[0, input_len:total_len]
                gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                # If generation produced nothing meaningful, fall back to decoding the entire output
                if not gen_text:
                    gen_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                return gen_text
            else:
                # No new tokens were produced (likely padding/truncation issues). Decode full output as fallback.
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        response = _single_generate(inputs)

        # Check if response is garbled or nonsensical
        def _is_garbled(text):
            if not text or len(text.strip()) < 10:
                return True
            # Check for repeated characters or nonsense patterns
            if re.search(r'(.)\1{3,}', text) or re.search(r'[A-Za-z]{2,}[A-Za-z]{2,}[A-Za-z]{2,}', text):
                return True
            # Check for very short words repeated
            words = text.split()
            if len(words) < 5 or any(len(w) < 2 for w in words[:3]):
                return True
            return False

        # If response is garbled, use fallback responses
        if response is None or _is_garbled(response):
            print("[Generation] Detected garbled response, using fallback...")
            
            # Topic-specific fallback responses with more precise matching
            prompt_lower = prompt.lower()
            if re.search(r'\b(penicillin|antibiotic|pharmaceutical|drug|medication)\b', prompt_lower):
                response = "I focus on natural health approaches rather than pharmaceutical medications. For questions about specific drugs like penicillin, I'd recommend consulting with your healthcare provider or pharmacist who can give you accurate medical information. Is there a natural health topic I can help you with instead?"
            elif re.search(r'\b(who\s+are\s+you|what\s+are\s+you|introduce\s+yourself)\b', prompt_lower):
                response = "Hi there! I'm Odin, your friendly AI assistant who loves chatting about natural health and wellness. I'm here to share information about herbs, nutrition, and holistic approaches to feeling your best. What would you like to explore today?"
            elif re.search(r'\b(stress|anxiety|calm|relax|stressed|anxious)\b', prompt_lower):
                response = "Great question about stress! Some wonderful natural approaches include herbs like ashwagandha and chamomile, breathing exercises, meditation, and regular movement. Each person responds differently, so it's about finding what works for you. What kind of stress relief are you most interested in exploring?"
            elif re.search(r'\b(energy|tired|fatigue|exhausted|boost)\b', prompt_lower):
                response = "For natural energy support, I love talking about things like B vitamins, adaptogenic herbs like rhodiola, good sleep hygiene, and balanced nutrition. Sometimes fatigue is our body's way of telling us something needs attention. What's your energy like throughout the day?"
            elif re.search(r'\b(natural\s+medicine|how\s+does.*work|healing|holistic)\b', prompt_lower):
                response = "Natural medicine is fascinating! It works by supporting your body's incredible ability to heal itself. Think of it as giving your body the right tools - whether that's nutrients, herbs, or lifestyle changes - to do what it naturally knows how to do. It focuses on treating root causes rather than just symptoms, and looks at the whole person. What aspect of natural healing interests you most?"
            elif re.search(r'\b(herb|plant|botanical|herbal)\b', prompt_lower):
                response = "I love talking about herbs! They're like nature's pharmacy - each one has its own unique properties and benefits. Some are great for relaxation, others for energy, digestion, or immune support. The key is finding the right herbs for your specific needs and using them safely. What kind of herbal support are you looking for?"
            else:
                response = "I'd love to help you with that! Natural health is such a rich topic. Could you tell me a bit more about what specific aspect you're curious about? I'm here to chat about herbs, nutrition, wellness practices, or any other natural health questions you might have."

        # If generation failed (e.g., CUDA assertion), try a CPU fallback then a concise canned answer
        elif response is None:
            print("[Generation] Primary generation failed; attempting CPU fallback for more stable generation...")
            # Attempt a CPU-only generation as a last-resort (may be slow for large models)
            try:
                if str(self.device).startswith('cuda'):
                    orig_device = self.device
                    try:
                        # Move model to CPU temporarily for stable generation
                        self.model = self.model.to(torch.device('cpu'))
                        self.device = torch.device('cpu')
                        inputs_cpu = {k: v.cpu() for k, v in inputs.items()}
                        with torch.no_grad():
                            cpu_out = self.model.generate(
                                **inputs_cpu,
                                max_new_tokens=max_length,
                                do_sample=True,
                                top_p=0.9,
                                temperature=0.8,
                                num_return_sequences=1,
                                repetition_penalty=1.2,
                                no_repeat_ngram_size=3
                            )
                        try:
                            if 'attention_mask' in inputs_cpu:
                                input_len = int(inputs_cpu['attention_mask'].sum().item())
                            else:
                                input_len = inputs_cpu['input_ids'].ne(self.tokenizer.pad_token_id).sum().item()
                        except Exception:
                            input_len = cpu_out.shape[-1]
                        if input_len < cpu_out.shape[-1]:
                            response = self.tokenizer.decode(cpu_out[0, input_len:], skip_special_tokens=True).strip()
                        else:
                            response = self.tokenizer.decode(cpu_out[0], skip_special_tokens=True).strip()
                    finally:
                        # Try to move model back to original device; if it fails, leave it on CPU
                        try:
                            if orig_device is not None and str(orig_device).startswith('cuda'):
                                self.model = self.model.to(orig_device)
                                self.device = orig_device
                        except Exception:
                            # If moving back fails, keep model on CPU but update device
                            self.device = torch.device('cpu')
                else:
                    # Already on CPU, nothing else to try
                    response = None
            except Exception as e:
                print(f"[Generation] CPU fallback also failed: {e}")

            if response is None:
                print("[Generation] CPU fallback did not succeed; returning concise fallback summary.")
                # Use the first few-shot example as a safe fallback summary
                try:
                    response = few_shot_examples[0][1]
                except Exception:
                    response = "Natural medicine is an approach that uses natural agents and traditional practices to support health. Consult qualified practitioners and your healthcare provider for advice."

        # If the beam output looks like gibberish (very short, lots of non-word chars,
        # or repeated characters), retry with sampling. If sampling still fails,
        # fall back to a small deterministic reply for identity questions.
        def _looks_gibberish(s: str) -> bool:
            if not s:
                return True
            s = s.strip()
            # Too short (only punctuation or a few chars)
            if len(re.findall(r"\w", s)) < 5:
                return True
            # Too many non-word chars compared to total length
            non_word_ratio = len(re.findall(r"[^\w\s]", s)) / max(1, len(s))
            if non_word_ratio > 0.4:
                return True
            # Repeated character sequences like 'aaaaaa' or '.....'
            if re.search(r"(.)\\1{4,}", s):
                return True
            return False

        if _looks_gibberish(response):
            # Try a sampling-based retry with a higher token budget and temperature
            try:
                with torch.no_grad():
                    sample_outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        do_sample=True,
                        top_p=0.95,
                        temperature=0.7,
                        num_return_sequences=1,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=3
                    )
                # decode only newly generated tokens
                try:
                    if 'attention_mask' in inputs:
                        input_len = int(inputs['attention_mask'].sum().item())
                    else:
                        input_len = inputs['input_ids'].ne(self.tokenizer.pad_token_id).sum().item()
                except Exception:
                    input_len = sample_outputs.shape[-1]
                if input_len < sample_outputs.shape[-1]:
                    sample_text = self.tokenizer.decode(sample_outputs[0, input_len:], skip_special_tokens=True).strip()
                else:
                    sample_text = self.tokenizer.decode(sample_outputs[0], skip_special_tokens=True).strip()
                if not _looks_gibberish(sample_text):
                    response = sample_text
                else:
                    # If sampling also produced gibberish, try generating with the original base model
                    try:
                        if hasattr(self, 'base_model_name') and self.base_model_name and (hasattr(self, '_base_model') is False or self._base_model is None):
                            # Load base model lazily and cache it
                            try:
                                self._base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
                                self._base_model.to(self.device)
                                # Ensure use_cache False if gradient checkpointing is on
                                try:
                                    self._base_model.config.use_cache = False
                                except Exception:
                                    pass
                            except Exception:
                                self._base_model = None

                        if getattr(self, '_base_model', None) is not None:
                            with torch.no_grad():
                                base_outs = self._base_model.generate(
                                    **inputs,
                                    max_new_tokens=max_length,
                                    do_sample=True,
                                    temperature=0.7,
                                    top_p=0.92,
                                    repetition_penalty=1.05,
                                    no_repeat_ngram_size=3,
                                    num_return_sequences=1
                                )
                            try:
                                if 'attention_mask' in inputs:
                                    input_len = int(inputs['attention_mask'].sum().item())
                                else:
                                    input_len = inputs['input_ids'].ne(self.tokenizer.pad_token_id).sum().item()
                            except Exception:
                                input_len = base_outs.shape[-1]
                            if input_len < base_outs.shape[-1]:
                                base_text = self.tokenizer.decode(base_outs[0, input_len:], skip_special_tokens=True).strip()
                            else:
                                base_text = self.tokenizer.decode(base_outs[0], skip_special_tokens=True).strip()
                            if not _looks_gibberish(base_text):
                                response = base_text
                                # fall through to cleaning
                            else:
                                response = None
                        else:
                            response = None
                    except Exception:
                        response = None

                    if not response:
                        # Final fallback: ask for clarification rather than returning garbage
                        if re.search(r"who\s+are\s+you", prompt, flags=re.I):
                            response = "Hey there! I'm Odin, your friendly AI assistant who loves chatting about natural health and wellness. What's on your mind?"
                        else:
                            fallback_responses = [
                                "Hmm, I'm having a bit of trouble with that one. Could you try rephrasing your question? I'd love to help!",
                                "You know what, let me think about that differently. Could you give me a bit more context about what you're looking for?",
                                "I'm not quite getting the right words together for that. Mind if we try approaching it from another angle?",
                                "Oops, seems like I got a bit tangled up there! Could you help me out by asking that in a different way?",
                                "I want to give you a really good answer, but I'm struggling with that one. What specific aspect interests you most?"
                            ]
                            response = random.choice(fallback_responses)
            except Exception:
                # If generation retry fails for any reason, leave the original response
                pass

        # Clean up and format the response
        response = response.strip().replace("[SEP]", "").replace("[CLS]", "")

        # Remove instruction leakage and clean up response
        def _clean_text(text: str) -> str:
            text = text.strip()
            
            # Remove instruction patterns that might leak through
            instruction_patterns = [
                r"Answer this question about natural health.*?:",
                r"Please respond naturally as Odin.*?:",
                r"Keep your response:.*?Response:",
                r"- Warm and engaging.*?- Reference previous conversation topics when relevant",
                r"Warm and engaging, like talking to a friend.*?Reference previous conversation topics when relevant",
                r"My friendly response:\s*",
                r"Question:.*?My friendly response:",
                r"I am Odin.*?My friendly response:",
                r"Response:\s*",
                r"Answer:\s*"
            ]
            
            for pattern in instruction_patterns:
                text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
            
            # Clean up bullet points and formatting artifacts
            text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)
            text = re.sub(r'\n\s*[-•]\s*', ' ', text)
            
            # Fix common garbled patterns
            text = re.sub(r'C\s*can\s*you\s*tell\s*me\s*about', 'Can you tell me about', text, flags=re.IGNORECASE)
            text = re.sub(r'penen?ec[ai]l?in', 'penicillin', text, flags=re.IGNORECASE)
            text = re.sub(r'\(Periodic\):\s*Yes\.?\s*', '', text)
            
            # Remove repeated question patterns
            text = re.sub(r'Can you tell.*?what is it\?', '', text, flags=re.IGNORECASE)
            
            # Only collapse excessive repeated characters (3+ times)
            text = re.sub(r'(.)\1{3,}', r'\1', text)
            # Preserve natural punctuation but collapse excessive sequences (3+ times)
            text = re.sub(r'[\.\,\!\?]{3,}', lambda m: m.group(0)[:2], text)
            # Only collapse words repeated 3+ times
            text = re.sub(r"\b(\w+)(?:\s+\1){3,}\b", r"\1", text, flags=re.I)
            
            # Allow longer responses but cap at 4 sentences for natural conversation flow
            sentences = re.split(r'(?<=[\.!?])\s+', text)
            if len(sentences) > 4:
                text = ' '.join(sentences[:4])
            
            # Clean up extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # If text is still garbled or too short, return None to trigger fallback
            if len(text.strip()) < 20 or not re.search(r'[a-zA-Z]{3,}', text):
                return None
            
            return text.strip()

        cleaned_response = _clean_text(response)
        if cleaned_response is None:
            # Text was too garbled, use fallback
            prompt_lower = prompt.lower()
            if re.search(r'\b(penicillin|antibiotic|pharmaceutical)\b', prompt_lower):
                response = "I focus on natural health approaches rather than pharmaceutical medications. For questions about specific drugs like penicillin, I'd recommend consulting with your healthcare provider. Is there a natural health topic I can help you with instead?"
            elif re.search(r'\b(natural\s+medicine|how\s+does.*work)\b', prompt_lower):
                response = "Natural medicine works by supporting your body's amazing ability to heal itself! It uses things like herbs, nutrition, and lifestyle changes to address root causes and promote overall wellness. What aspect interests you most?"
            else:
                response = "I'd love to help you with that natural health question! Could you rephrase it a bit? I'm here to chat about herbs, nutrition, wellness practices, and other natural approaches to health."
        else:
            response = cleaned_response

        # Strip any "User:" or "Assistant:" prefixes
        response = re.sub(r"^(User|Assistant):\s*", "", response, flags=re.I)

        # Add natural, varied conversational endings
        import random
        natural_endings = [
            "What do you think? Does that help clarify things?",
            "Hope that gives you a good starting point! What else would you like to explore?",
            "Let me know if you'd like me to dive deeper into any part of that!",
            "Does that answer what you were wondering about?",
            "I'd love to hear your thoughts on this - what resonates with you?",
            "Feel free to ask if you want to explore any of this further!",
            "What other questions are on your mind?",
            "Hope that's helpful! I'm here if you want to chat more about it.",
            "Curious to know - is there a particular aspect you're most interested in?",
            "Let me know what other questions come up for you!"
        ]
        
        # Only add endings to substantial responses, and vary them naturally
        if len(response.strip()) > 50 and not re.search(r'[?!]$', response.strip()):
            if random.random() < 0.7:  # 70% chance to add an ending
                ending = random.choice(natural_endings)
                response += f" {ending}"
        elif len(response.strip()) <= 50:
            response += " Hope that helps!"

        # If requested, ensure the response is relevant to the prompt using a simple keyword overlap heuristic.
        if ensure_relevant:
            # Build keyword set from prompt: words longer than 3 chars excluding a small stoplist
            stopwords = {"the","and","for","with","that","this","from","what","when","where","how","why","which","your","you","are","was","were","will","have","has","had"}
            prompt_words = set([w.lower() for w in re.findall(r"\w+", prompt) if len(w) > 3 and w.lower() not in stopwords])
            attempt = 1
            # simple check: at least one keyword from prompt appears in response
            while attempt <= max_attempts:
                resp_text = response.lower()
                overlap = any(k in resp_text for k in prompt_words) if prompt_words else True
                if overlap:
                    break
                # otherwise, retry generation with a clearer instruction and lower temperature
                extra = "\n\nPlease answer the user's question directly and include the key points from the prompt. Be concise."
                inputs_retry = self.tokenizer(prompt + extra, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
                # ensure we pass the new tokenized inputs to the generator
                response = _single_generate(inputs_retry, temp=0.5)
                attempt += 1

        # Auto-tune generation parameters based on response quality
        self._auto_tune_parameters(response, prompt)
        
        # Return structured JSON by default to make it easy for downstream systems to parse.
        if response_format == "json":
            resp_obj = {
                "message": response,
                "metadata": {
                    "length": len(response),
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    "model": self.model_name,
                    "device": str(self.device),
                    "mood": self.conversation_mood,
                    "interests": list(self.user_interests) if self.user_interests else []
                }
            }
            try:
                return json.dumps(resp_obj, ensure_ascii=False)
            except Exception:
                return json.dumps(resp_obj)

        return response
    
    def _auto_tune_parameters(self, response, prompt):
        """Auto-tune generation parameters based on response quality"""
        # Simple quality metrics
        response_length = len(response.split())
        has_questions = '?' in response
        has_enthusiasm = '!' in response
        seems_natural = any(word in response.lower() for word in ['you know', 'i think', 'actually', 'really', 'pretty', 'quite'])
        
        # Adjust temperature based on response characteristics
        if not seems_natural and not has_enthusiasm:
            # Response seems too formal, increase temperature slightly
            if hasattr(self, '_current_temperature'):
                self._current_temperature = min(0.9, self._current_temperature + 0.05)
            else:
                self._current_temperature = 0.85
        elif response_length < 20:
            # Response too short, encourage more generation
            if hasattr(self, '_current_min_length'):
                self._current_min_length = min(120, self._current_min_length + 10)
            else:
                self._current_min_length = 90
        
        # Track conversation depth for context awareness
        if any(word in prompt.lower() for word in ['tell me more', 'explain', 'how', 'why', 'what about']):
            self.conversation_depth += 1
        else:
            self.conversation_depth = max(0, self.conversation_depth - 1)

    def prepare_dataset(self, dataset_name="nikhilkeetha/personalized-assistant", split="train"):
        """Load and preprocess dataset for training"""
        print(f"Loading dataset {dataset_name}...")
        try:
            # Try to load the requested dataset from the Hub
            dataset = load_dataset(dataset_name, split=split)

            # Take a very small subset for initial testing (fast smoke test)
            dataset = dataset.shuffle(seed=42).select(range(20))  # Start with 20 examples
            print(f"Selected {len(dataset)} examples for training")

            def process_conversations(example):
                # Create conversational format from question and answer
                # Support common schema (squad-like) or fallback to generic fields
                if "question" in example and "answers" in example:
                    answer_text = example["answers"]["text"][0] if isinstance(example["answers"], dict) and example["answers"].get("text") else (example.get("answer") or "")
                    conversation = f"User: {example['question']}\nAssistant: {answer_text}"
                elif "text" in example:
                    conversation = example["text"]
                else:
                    # Try to join string fields
                    vals = [str(v) for v in example.values() if isinstance(v, (str, int, float))]
                    conversation = " ".join(vals)
                return {"text": conversation.strip()}

            print("Processing dataset...")
            # Process dataset with progress bar (single-process to avoid Windows pickling issues)
            # Use an environment-aware number of workers: Windows multiprocessing with
            # fork/spawn can hit pickling issues; use 1 worker on Windows, 3 elsewhere.
            is_windows = platform.system().lower().startswith("win")
            num_proc = 1 if is_windows else 3
            print(f"Processing dataset with num_proc={num_proc} (Windows detection: {is_windows})")
            processed_dataset = dataset.map(
                process_conversations,
                desc="Converting to conversation format",
                num_proc=num_proc
            )
            print("Dataset processing completed")
            
        except Exception as e:
            print(f"Error loading dataset '{dataset_name}': {str(e)}")
            # Detect common multiprocessing pickler issue in some Windows/Python setups
            if "Pickler._batch_setitems" in str(e) or "Pickler._batch_setitems()" in str(e):
                print("Detected Pickler multiprocessing issue. Falling back to local small dataset.")
                # Try to build a tiny dataset from local `user_data.jsonl` if present
                fallback_texts = []
                try:
                    if os.path.exists("user_data.jsonl"):
                        with open("user_data.jsonl", "r", encoding="utf-8") as uf:
                            for line in uf:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    obj = json.loads(line)
                                    if isinstance(obj, dict) and obj.get("user") and obj.get("assistant"):
                                        fallback_texts.append(f"User: {obj['user']}\nAssistant: {obj['assistant']}")
                                except Exception:
                                    continue
                except Exception:
                    pass

                if not fallback_texts:
                    # Small synthetic fallback so training can continue
                    fallback_texts = [
                        "User: Who are you?\nAssistant: I am Odin, an all-knowing but friendly assistant.",
                        "User: How are you?\nAssistant: I'm doing well — ready to help."
                    ]

                # Avoid constructing a Hugging Face Dataset here (can trigger pickling issues on Windows).
                processed_dataset = [{"text": t} for t in fallback_texts]
                print(f"Using fallback local dataset with {len(processed_dataset)} examples.")
            else:
                # For other errors, re-raise to make the failure visible
                print("Attempting a simple 'squad' fallback...")
                try:
                    dataset = load_dataset("squad", split=split)
                    dataset = dataset.shuffle(seed=42).select(range(20))
                    def process_conversations2(example):
                        conversation = f"User: {example['question']}\nAssistant: {example['answers']['text'][0]}"
                        return {"text": conversation.strip()}
                    # Fallback: be environment-aware here too
                    is_windows = platform.system().lower().startswith("win")
                    num_proc = 1 if is_windows else 3
                    processed_dataset = dataset.map(process_conversations2, num_proc=num_proc)
                    print("Fallback 'squad' dataset loaded successfully.")
                except Exception as e3:
                    print(f"Error loading fallback dataset: {e3}")
                    raise e
        
        def _tokenize_texts(texts):
            # Tokenize a list of raw texts and return a torch TensorDataset
            processed = [self.tokenizer(
                t,
                truncation=True,
                padding="max_length",
                max_length=512
            ) for t in texts]

            input_ids = torch.stack([torch.tensor(d["input_ids"]) for d in processed])
            attention_mask = torch.stack([torch.tensor(d["attention_mask"]) for d in processed])
            # Return a dataset that yields dicts so DataCollatorForLanguageModeling can process samples
            class DictDataset(torch.utils.data.Dataset):
                def __init__(self, input_ids, attention_mask):
                    self.input_ids = input_ids
                    self.attention_mask = attention_mask
                def __len__(self):
                    return self.input_ids.size(0)
                def __getitem__(self, idx):
                    return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx]}

            return DictDataset(input_ids, attention_mask)

        # If processed_dataset is a plain list (fallback) or a HF Dataset, handle both cases.
        try:
            # HF dataset-like object with column access
            if hasattr(processed_dataset, "column_names"):
                # Extract texts safely
                if "text" in processed_dataset.column_names:
                    texts = [x for x in processed_dataset["text"]]
                else:
                    # Build texts by joining available string fields
                    texts = []
                    for ex in processed_dataset:
                        if isinstance(ex, dict) and ex.get("text"):
                            texts.append(ex["text"])
                        else:
                            vals = [str(v) for v in ex.values() if isinstance(v, (str, int, float))]
                            texts.append(" ".join(vals))
            else:
                # Assume it's a list of dicts
                texts = [d.get("text", "") for d in processed_dataset]
        except Exception:
            # Last resort: iterate to build texts
            texts = []
            try:
                for ex in processed_dataset:
                    if isinstance(ex, dict) and ex.get("text"):
                        texts.append(ex["text"])
                    else:
                        vals = [str(v) for v in ex.values() if isinstance(v, (str, int, float))]
                        texts.append(" ".join(vals))
            except Exception:
                texts = ["User: Hello\nAssistant: Hi there!"]

        tokenized_dataset = _tokenize_texts(texts)
        return tokenized_dataset

    def fine_tune(self, dataset_name="daily_dialog", output_dir="./odin-finetuned"):
        """Fine-tune Odin on conversational dataset"""
        print("Preparing dataset...")
        train_dataset = self.prepare_dataset(dataset_name)
        dataset_size = len(train_dataset)
        print(f"Dataset size: {dataset_size} examples")
        
        # Calculate optimal batch size based on available memory
        batch_size = 1 if torch.cuda.is_available() else 1
        
        print("Configuring training arguments based on hardware...")
        # Decide fp16 and batch sizing heuristics (RTX 2060 ~6GB VRAM)
        gpu_mem = self.hardware.get("gpu_total_mem_mb") or 0
        system_ram = self.hardware.get("system_ram_gb") or 0
        use_fp16 = False
        per_device_batch = 1
        gradient_accumulation = 1
        learning_rate = 5e-5
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

        # Safety overrides for small datasets to avoid fp16/grad explosion
        if dataset_size < 100:
            # Use safer, deterministic settings for small datasets
            use_fp16 = False
            per_device_batch = 1
            gradient_accumulation = 1
            # Lower the learning rate for stability
            learning_rate = min(learning_rate, 1e-5)
            print("Small dataset (<100) detected: forcing fp16=False, LR<=1e-5, and smaller accumulation to avoid instability.")

        # Very tiny datasets get even more conservative settings
        if dataset_size < 16:
            learning_rate = min(learning_rate, 5e-6)
            print("Very small dataset (<16): further lowering learning rate to reduce gradient explosion risk.")

        # Choose number of epochs: use more epochs for small datasets so the model can see
        # the tiny signal multiple times (conservative default: 3 epochs for <100 examples)
        num_epochs = 3 if dataset_size < 100 else 1
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_batch,
            gradient_accumulation_steps=gradient_accumulation,
            save_steps=10,
            save_total_limit=2,
            logging_steps=1,
            fp16=use_fp16,
            learning_rate=learning_rate,
            max_grad_norm=1.0,
            dataloader_pin_memory=bool(torch.cuda.is_available()),
            logging_dir=f"{output_dir}/logs"
        )
        print(f"Training config: fp16={use_fp16}, batch={per_device_batch}, accum={gradient_accumulation}, epochs={num_epochs}")
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset
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

    # Parse simple CLI flags
    use_finetuned = False
    test_finetuned_prompt = None
    no_finetune = False
    quick = False
    force_cpu = False
    spawn_continual = True

    if "--use-finetuned" in sys.argv:
        use_finetuned = True
    if "--test-finetuned" in sys.argv:
        # Next arg may be the prompt; otherwise use default
        idx = sys.argv.index("--test-finetuned")
        if len(sys.argv) > idx + 1:
            test_finetuned_prompt = sys.argv[idx + 1]
    if "--no-finetune" in sys.argv or "--skip-finetune" in sys.argv:
        no_finetune = True
    if "--quick" in sys.argv:
        quick = True
    if "--cpu" in sys.argv or "--force-cpu" in sys.argv:
        force_cpu = True
    if "--no-spawn-continual" in sys.argv or "--no-continual" in sys.argv:
        spawn_continual = False

    # Initialize Odin
    model_dir = os.path.abspath("./odin-finetuned") if use_finetuned else None
    # Temporarily filter noisy advisory lines during model/tokenizer loading and related imports.
    _saved_out_main, _saved_err_main = sys.stdout, sys.stderr
    patterns = [r"high_res_pixel_values", r"custom_generate/generate.py", r"Config not found for parakeet"]
    fs = _FilteredStream(_saved_out_main, block_patterns=patterns)
    sys.stdout = fs
    sys.stderr = fs
    try:
        odin = OdinAssistant(finetuned_dir=model_dir, force_cpu=force_cpu) if model_dir else OdinAssistant(force_cpu=force_cpu)
    finally:
        sys.stdout, sys.stderr = _saved_out_main, _saved_err_main
    # If user requested CPU-only operation, force device
    if force_cpu:
        try:
            odin.model = odin.model.to(torch.device('cpu'))
            odin.device = torch.device('cpu')
            print("[CLI] Forced CPU mode: model moved to CPU for stable generation.")
        except Exception as e:
            print(f"[CLI] Could not force CPU mode: {e}")

    # Fine-tune on each dataset (initial training) unless user opted out
    if not no_finetune and not quick:
        for dataset_name in dataset_names:
            print(f"\n--- Fine-tuning on dataset: {dataset_name} ---")
            try:
                odin.fine_tune(dataset_name=dataset_name)
            except Exception as e:
                print(f"Failed to fine-tune on {dataset_name}: {e}")
    else:
        print("[CLI] Skipping fine-tuning (flags: no_finetune/quick set or quick mode). Entering interactive REPL.")

    # Optionally start continual training in a new terminal (only if spawn_continual is True)
    if spawn_continual:
        try:
            import subprocess
            python_exe = sys.executable
            script_path = os.path.abspath(__file__)
            # Use Windows 'start' to open a new cmd window running the continual train flag
            subprocess.Popen([
                "start", "cmd", "/k", f"{python_exe} {script_path} --continual-train"
            ], shell=True)
            print("[CLI] Spawned continual trainer in a new terminal.")
        except Exception as e:
            print(f"[CLI] Could not spawn continual trainer: {e}")
    else:
        print("[CLI] Not spawning continual trainer (flag set).")

    print("\n🌿 Hey there! I'm Odin, your friendly AI companion for all things natural health and wellness!")
    print("I love having real conversations and learning about what interests you.")
    print("Feel free to ask me anything - from herbs and nutrition to wellness practices and beyond.")
    print("Type 'exit' when you're ready to go. Let's chat! 😊\n")
    conversation_history = []

    # If test_finetuned_prompt is provided, run a single-shot test using the finetuned model
    if test_finetuned_prompt is not None:
        prompt = test_finetuned_prompt
        print(f"\n[Finetuned test] Prompt: {prompt}")
        response = odin.generate_response(prompt, response_format="text", ensure_relevant=True)
        print(f"[Finetuned test] Response:\n{response}")
        return

    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Odin: Farewell! Until next time.")
                break

            # Build natural conversation context with recent history
            if conversation_history:
                # Include last 3 exchanges for better context, but keep it natural
                recent_context = "\n".join(conversation_history[-6:])  # Last 3 exchanges
                # Create a more natural prompt that flows like a real conversation
                full_prompt = f"Previous conversation:\n{recent_context}\n\nUser: {user_input}\n\nPlease respond naturally as Odin:"
            else:
                # First message - just use the user input with natural framing
                full_prompt = f"User: {user_input}\n\nPlease respond naturally as Odin:"

            # For interactive use, request plain text so we don't store raw JSON into history
            response_text = odin.generate_response(full_prompt, response_format="text")
            print(f"Odin: {response_text}")

            # Update conversation history with plain text assistant reply
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Assistant: {response_text}")

            # Add user interaction for continual training (store text)
            odin.add_user_example(user_input, response_text)
    except KeyboardInterrupt:
        print("\nInterrupted — exiting gracefully.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--continual-train":
        OdinAssistant.continual_fine_tune_process()
    else:
        main()

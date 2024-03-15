from transformers import LlamaConfig, TrainingArguments


MODEL_CONFIG = LlamaConfig(
    vocab_size=50257,
    hidden_size=128,
    num_hidden_layers=8,
    intermediate_size=256,
    num_attention_heads=8,
    max_position_embeddings=256,
    bos_token_id=50256,
    eos_token_id=50256
)


TRAINING_ARGS = TrainingArguments(
    output_dir="/proj/mechanistic.shadow/mrofin/tinylinguist/models",
    evaluation_strategy="steps",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    gradient_accumulation_steps=8,
    learning_rate=1e-3,
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    num_train_epochs=5,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    logging_first_step=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=None,
    seed=42,
    fp16=True,
    eval_steps=200,
    run_name="test_run",
    group_by_length=True,
    report_to="wandb",
    remove_unused_columns=True
)

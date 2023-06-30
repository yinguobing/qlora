import evaluate
import torch
import numpy as np
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaTokenizer, set_seed

from .callback import SavePeftModelCallback
from .dataset import IGNORE_INDEX, make_data_module, make_mmlu_dataset
from .model import (
    DEFAULT_PAD_TOKEN,
    get_accelerate_model,
    get_last_checkpoint,
    print_trainable_parameters,
    smart_tokenizer_and_embedding_resize,
)


def make_trainer(args, training_args):
    # Load the model from checkpoint
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print("Detected that training was already completed!")
    model = get_accelerate_model(args, checkpoint_dir)
    model.config.use_cache = False
    print_trainable_parameters(args, model)
    print(f"Model loaded from {checkpoint_dir}")
    set_seed(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False,  # Fast tokenizer giving issues.
        tokenizer_type="llama"
        if "llama" in args.model_name_or_path
        else None,  # Needed for HF name change
        use_auth_token=args.use_auth_token,
    )
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print("Adding special tokens.")
        tokenizer.add_special_tokens(
            {
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id
                    if model.config.pad_token_id != -1
                    else tokenizer.pad_token_id
                ),
            }
        )

    # Data
    data_module = make_data_module(tokenizer=tokenizer, args=args)

    # Trainer
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},
    )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)

    if args.do_mmlu_eval:
        mmlu_dataset = make_mmlu_dataset(args)
        abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]
        accuracy = evaluate.load("accuracy")

        class MMLUEvalCallback(transformers.TrainerCallback):
            def on_evaluate(self, args, state, control, model, **kwargs):
                data_loader = trainer.get_eval_dataloader(mmlu_dataset)
                source_max_len = trainer.data_collator.source_max_len
                trainer.data_collator.source_max_len = args.mmlu_source_max_len
                trainer.model.eval()
                preds, refs = [], []
                loss_mmlu = 0
                for batch in tqdm(data_loader, total=len(data_loader)):
                    (loss, logits, labels) = trainer.prediction_step(
                        trainer.model,
                        batch,
                        prediction_loss_only=False,
                    )
                    # There are two tokens, the output, and eos token.
                    for i, logit in enumerate(logits):
                        label_non_zero_id = (batch["labels"][i] != -100).nonzero()[0][0]
                        logit_abcd = logit[label_non_zero_id - 1][abcd_idx]
                        preds.append(torch.argmax(logit_abcd).item())
                    labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:, 0]
                    refs += [abcd_idx.index(label) for label in labels.tolist()]
                    loss_mmlu += loss.item()
                # Extract results by subject.
                results = {"mmlu_loss": loss_mmlu / len(data_loader)}
                subject = mmlu_dataset["subject"]
                subjects = {s: {"refs": [], "preds": []} for s in set(subject)}
                for s, p, r in zip(subject, preds, refs):
                    subjects[s]["preds"].append(p)
                    subjects[s]["refs"].append(r)
                subject_scores = []
                for subject in subjects:
                    subject_score = accuracy.compute(
                        references=subjects[subject]["refs"],
                        predictions=subjects[subject]["preds"],
                    )["accuracy"]
                    results[
                        f"mmlu_{args.mmlu_split}_accuracy_{subject}"
                    ] = subject_score
                    subject_scores.append(subject_score)
                results[f"mmlu_{args.mmlu_split}_accuracy"] = np.mean(subject_scores)
                trainer.log(results)
                trainer.data_collator.source_max_len = source_max_len

        trainer.add_callback(MMLUEvalCallback)

    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    return trainer, tokenizer, data_module

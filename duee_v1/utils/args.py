import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a task (SPO) with accelerate library"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lenght` is passed."
        ),
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="./data",
        help="file_path.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="hfl/chinese-roberta-wwm-ext",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.02, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=20,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps_or_radios",
        type=eval,
        default=0.1,
        help="Number of steps or radios for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/",
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data_caches",
        help="Where to store data caches.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert",
        help="Model type to use.",
        choices=["roformer", "bert", "chinesebert"],
    )

    parser.add_argument(
        "--writer_type",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "visualdl"],
        help="writer_type",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=200,
        help="logging_steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10804,
        help="save_steps.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=1,
        help="save_topk.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_workers",
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--model_cache_dir", default=None, type=str, help="Huggingface model cache_dir"
    )

    parser.add_argument(
        "--use_efficient", action="store_true", help="use EfficientGlobalPointer"
    )

    args = parser.parse_args()

    if args.output_dir is not None:
        model_weights = args.pretrained_model_name_or_path.replace("/", "_").replace(
            ":", ""
        )
        args.output_dir = os.path.join(
            args.output_dir,
            f"{args.model_type}-{model_weights}",
        )
        args.model_weights = model_weights
        args.log_dir = os.path.join(args.output_dir, "logs")
        os.makedirs(args.log_dir, exist_ok=True)

    return args

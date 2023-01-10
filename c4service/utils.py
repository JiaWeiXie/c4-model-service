from typing import List, Optional, Tuple

from transformers import RobertaTokenizer


def init_tokenizer(
    model_name: Optional[str] = None,
    do_lower_case: bool = True,
) -> RobertaTokenizer:
    default_model = "microsoft/graphcodebert-base"
    # default_model = "microsoft/codebert-base"

    return RobertaTokenizer.from_pretrained(
        model_name if model_name else default_model,
        do_lower_case=do_lower_case,
    )


def source_process(
    source: str,
    max_source_length: int = 512,
    tokenizer: Optional[RobertaTokenizer] = None,
    model_name: Optional[str] = None,
) -> Tuple[List[int], List[int]]:
    if not tokenizer:
        tokenizer = init_tokenizer(model_name)

    source_tokens = tokenizer.tokenize(source)[: max_source_length - 2]
    source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    source_mask = [1] * (len(source_tokens))
    padding_length = max_source_length - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    source_mask += [0] * padding_length
    return source_ids, source_mask


def target_process(
    target: str,
    max_target_length: int = 512,
) -> Tuple[List[int], List[int]]:
    max_length = max_target_length - 2
    target_tokens = target[:max_length]
    target_ids = [int(target_tokens)]
    target_mask = [1] * len(target_ids)
    return target_ids, target_mask

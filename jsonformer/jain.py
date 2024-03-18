import json
from typing import Any, Dict, List, Union

import torch
from termcolor import cprint
from transformers import PreTrainedModel, PreTrainedTokenizer

from jsonformer.logits_processors import (NumberStoppingCriteria,
                                          OutputNumbersTokens,
                                          StringStoppingCriteria)

GENERATION_MARKER = "|GENERATION|"

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


class Jsonformer:
    value: Dict[str, Any] = {}

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        json_schema: Dict[str, Any],
        prompt: str,
        images: torch.Tensor,
        cross_images: torch.Tensor,
        *,
        debug: bool = False,
        max_array_length: int = 10,
        max_number_tokens: int = 6,
        temperature: float = 1.0,
        max_string_token_length: int = 10,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.json_schema = json_schema
        self.prompt = prompt
        self.images = images
        self.cross_images = cross_images

        self.number_logit_processor = OutputNumbersTokens(self.tokenizer, self.prompt)

        self.generation_marker = "|GENERATION|"
        self.debug_on = debug
        self.max_array_length = max_array_length

        self.max_number_tokens = max_number_tokens
        self.temperature = temperature
        self.max_string_token_length = max_string_token_length

    def debug(self, caller: str, value: str, is_prompt: bool = False):
        if self.debug_on:
            if is_prompt:
                cprint(caller, "green", end=" ")
                cprint(value, "yellow")
            else:
                cprint(caller, "green", end=" ")
                cprint(value, "blue")

    def generate_number(self, temperature: Union[float, None] = None, iterations=0):
        model_inputs = self.get_inputs()
        # self.debug("[generate_number]", prompt, is_prompt=True)
        # input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
        #     self.model.device
        # )
        input_tokens = model_inputs["input_ids"]
        response = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_number_tokens,
            num_return_sequences=1,
            logits_processor=[self.number_logit_processor],
            stopping_criteria=[
                NumberStoppingCriteria(self.tokenizer, len(input_tokens[0]))
            ],
            temperature=temperature or self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(response[0], skip_special_tokens=True)

        response = response[len(prompt) :]
        response = response.strip().rstrip(".")
        self.debug("[generate_number]", response)
        try:
            return float(response)
        except ValueError:
            if iterations > 3:
                raise ValueError("Failed to generate a valid number")

            return self.generate_number(
                temperature=self.temperature * 1.3, iterations=iterations + 1
            )

    def generate_boolean(self) -> bool:
        model_inputs = self.get_inputs()
        # self.debug("[generate_boolean]", prompt, is_prompt=True)

        # input_tensor = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.forward(**model_inputs)
        logits = output.logits[0, -1]

        # todo: this assumes that "true" and "false" are both tokenized to a single token
        # this is probably not true for all tokenizers
        # this can be fixed by looking at only the first token of both "true" and "false"
        true_token_id = self.tokenizer.convert_tokens_to_ids("true")
        false_token_id = self.tokenizer.convert_tokens_to_ids("false")

        result = logits[true_token_id] > logits[false_token_id]

        self.debug("[generate_boolean]", result)

        return result.item()

    def generate_string(self) -> str:
        model_inputs = self.get_inputs(is_string=True)
        # self.debug("[generate_string]", prompt, is_prompt=True)
        # input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
        #     self.model.device
        # )
        input_tokens = model_inputs["input_ids"]

        response = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_string_token_length,
            num_return_sequences=1,
            temperature=self.temperature,
            stopping_criteria=[
                StringStoppingCriteria(self.tokenizer, len(input_tokens[0]))
            ],
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Some models output the prompt as part of the response
        # This removes the prompt from the response if it is present
        input_tokens = model_inputs["input_ids"]
        if (
            len(response[0]) >= len(input_tokens[0])
            and (response[0][: len(input_tokens[0])] == input_tokens).all()
        ):
            response = response[0][len(input_tokens[0]) :]
        if response.shape[0] == 1:
            response = response[0]

        response = self.tokenizer.decode(response, skip_special_tokens=True)

        self.debug("[generate_string]", "|" + response + "|")

        if response.count('"') < 1:
            return response

        return response.split('"')[0].strip()

    def generate_object(
        self, properties: Dict[str, Any], obj: Dict[str, Any]
    ) -> Dict[str, Any]:
        for key, schema in properties.items():
            self.debug("[generate_object] generating value for", key)
            obj[key] = self.generate_value(schema, obj, key)
        return obj

    def generate_value(
        self,
        schema: Dict[str, Any],
        obj: Union[Dict[str, Any], List[Any]],
        key: Union[str, None] = None,
    ) -> Any:
        schema_type = schema["type"]
        if schema_type == "number":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_number()
        elif schema_type == "boolean":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_boolean()
        elif schema_type == "string":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_string()
        elif schema_type == "array":
            new_array = []
            obj[key] = new_array
            return self.generate_array(schema["items"], new_array)
        elif schema_type == "object":
            new_obj = {}
            if key:
                obj[key] = new_obj
            else:
                obj.append(new_obj)
            return self.generate_object(schema["properties"], new_obj)
        else:
            raise ValueError(f"Unsupported schema type: {schema_type}")

    def generate_array(self, item_schema: Dict[str, Any], obj: Dict[str, Any]) -> list:
        for _ in range(self.max_array_length):
            # forces array to have at least one element
            element = self.generate_value(item_schema, obj)
            obj[-1] = element

            obj.append(self.generation_marker)
            model_inputs = self.get_inputs()
            obj.pop()
            # input_tensor = self.tokenizer.encode(input_prompt, return_tensors="pt")
            output = self.model.forward(**model_inputs)
            logits = output.logits[0, -1]

            top_indices = logits.topk(30).indices
            sorted_token_ids = top_indices[logits[top_indices].argsort(descending=True)]

            found_comma = False
            found_close_bracket = False

            for token_id in sorted_token_ids:
                decoded_token = self.tokenizer.decode(token_id)
                if "," in decoded_token:
                    found_comma = True
                    break
                if "]" in decoded_token:
                    found_close_bracket = True
                    break

            if found_close_bracket or not found_comma:
                break

        return obj

    def get_inputs(self, is_string: bool = False):
        template = """{prompt}\nOutput result in the following JSON schema format:\n{schema}\nResult: {progress}"""
        progress = json.dumps(self.value)
        gen_marker_index = progress.find(f'"{self.generation_marker}"')
        if gen_marker_index != -1:
            progress = progress[:gen_marker_index]
        else:
            raise ValueError("Failed to find generation marker")

        prompt = template.format(
            prompt=self.prompt,
            schema=json.dumps(self.json_schema),
            progress=progress,
        )
        if is_string:
            prompt += '"'

        image_size = 224  # taken from config.json
        patch_size = 14  # taken from config.json

        input_ids = [self.tokenizer.bos_token_id]
        token_type_ids = [LANGUAGE_TOKEN_TYPE]

        # vision
        images = self.images
        cross_images = self.cross_images

        # language
        vision_token_num = (image_size // patch_size) * (image_size // patch_size) + 2
        input_ids += [self.tokenizer.pad_token_id] * vision_token_num
        token_type_ids += [VISION_TOKEN_TYPE] * vision_token_num

        text_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        input_ids += text_ids
        token_type_ids += [LANGUAGE_TOKEN_TYPE] * len(text_ids)
        attention_mask = [1] * len(input_ids)

        input_ids += [self.tokenizer.eos_token_id]
        token_type_ids += [LANGUAGE_TOKEN_TYPE]
        attention_mask += [1]

        input_ids = torch.tensor(input_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            # "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "cross_attention_mask": torch.ones((1, 1, 1)),
            "images": images,
            "cross_images": cross_images,
        }

    def __call__(self) -> Dict[str, Any]:
        self.value = {}
        generated_data = self.generate_object(
            self.json_schema["properties"], self.value
        )
        return generated_data

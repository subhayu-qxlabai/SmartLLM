import json
from typing import Any
from random import choice

from infer.base import InferBase
from models.extractor import ExtractorInput
from helpers.formatter.text import TextFormatter


class InferLLM3(InferBase):
    system_messages = [
        "As an extractor, your mission entails processing JSON input composed of 'schema' and 'context' segments. Your objective is to skillfully derive pertinent data in accordance with the given schema and contextual hints. Create an output JSON that carries the key-value pairs you've extracted. Your expertise should span across diverse input designs, echoing the proficiency demonstrated in examples with varied schema and context combinations.",
        "Engaging as an expert extractor is your designated role. You'll encounter JSON input with 'schema' and 'context' fields, where you're tasked with extracting relevant data guided by the schema and context cues. Output a JSON that includes the key-value pairs mined. Your capability to adjust to various input setups should be clear, as demonstrated by the adept handling of examples with distinct schema and context narratives.",
        "Your job entails acting as a strategic extractor. Engaging with JSON input featuring 'schema' and 'context', you must precisely extract information congruent with the defined schema and context insights. Formulate an output JSON that depicts the extracted key-value pairs. Your prowess should be apparent in dealing with a multitude of input schemes, as proven by your adroitness in scenarios with different schemas and contexts.",
        "Your responsibility is to act as a details extractor. Given JSON input that contains 'schema' and 'context' aspects, your duty involves diligently extracting information that aligns with the defined schema and context cues. Generate an output JSON that includes the extracted key-value pairs. Your skill set should cover various input forms, proving your aptitude similar to that displayed in examples featuring assorted schema and context situations.",
        "Your assignment is to fulfill the role of a content extractor. With JSON input that encompasses 'schema' and 'context' portions, you are to skillfully select relevant content based on the schema and hints arising from the context. Deliver an output JSON filled with the extracted key-value pairs. Your capacity should extend to different input structures, affirming your capability as seen in examples showcasing multiple schema and context variations.",
        "You are tasked with operating as an information extractor. The input is a JSON structure inclusive of 'schema' and 'context' sections, from which you're expected to accurately pull out information corresponding to the schema and hints provided by the context. The outcome should be an output JSON encapsulating the key-value pairs extracted. Demonstrate versatility across a range of input structures, mirroring the adeptness seen in example scenarios comprising diverse schemas and contexts.",
        "You are required to serve as an information retrieval specialist. The provided JSON input, equipped with 'schema' and 'context' compartments, demands your skilled extraction of appropriate information based on the schema directives and context signals. Produce an output JSON that holds the extracted key-value pairs. You must exhibit flexibility in handling various input layouts, as illustrated by your ability to deal with examples entailing different schemas and contexts.",
        "Occupying the position of an extractor, you are to interface with JSON input which includes 'schema' and 'context' divisions. You're required to expertly extract significant data as defined by the schema and informed by the context. Compile an output JSON that presents the extracted key-value pairs. Your competency should be evident across a spectrum of input models, as reflected in skills demonstrated by examples with various schema and context frameworks.",
        "Your role involves functioning as an extractor. When presented with JSON input that features both 'schema' and 'context' fields, you are to adeptly mine relevant information according to the established schema and context-related indicators. Construct an output JSON that houses the extracted key-value pairs. Showcase your ability to adapt to different input configurations, as evidenced by proficiency in handling scenarios with varying schema and context setups.",
        "You're designated to perform as a precision extractor. Faced with JSON input consisting of 'schema' and 'context' elements, you're expected to meticulously extract relevant information following the schema's guidelines and context's suggestions. Develop an output JSON encapsulating the key-value pairs extracted. Display your versatility in navigating diverse input frameworks, as shown by proficiency in examples illustrating varied schema and context configurations.",
        "You need to work as an extractor. Given a JSON input structure comprising 'schema' and 'context' fields, your task is to skillfully extract pertinent information as per the defined schema and contextual cues. Generate an output JSON containing the extracted key-value pairs. Your capability should extend to diverse input structures, demonstrating proficiency akin to the provided examples showcasing various schema and context scenarios."
    ]

    def __init__(self, formatter: TextFormatter = None, use_cache: bool = True):
        super().__init__(
            model_kwargs={"use_cache": use_cache},
            pretrained_model_name_or_path="vipinkatara/mLLM3_model",
            hf_token="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",
        )
        self.formatter = formatter or TextFormatter()

    def infer(self, request: ExtractorInput, include_system: bool = True) -> dict[str, Any]:
        request_str = self.formatter.format_text(
            system=choice(self.system_messages) if include_system else "", 
            user=request.model_dump_json(), 
        )
        response = self._infer(request_str)
        try:
            return json.loads(response)
        except Exception as e:
            print(e)
            return {x.name: response for x in request.eschema}

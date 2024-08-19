TARGET_PROMPT = {
    "llama2": """[INST] {input} [/INST]""",
    "llama3": """<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n""",
}

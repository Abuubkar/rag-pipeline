# Step 5: structured output

Notes for [`step05_structured.py`](step05_structured.py).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| json_schema (Ollama's default) | Ollama forces the output to match the schema | Simple schemas; must never crash on format | The model never sees field descriptions: put the rules in the prompt (step 17) |
| ✅ function_calling | The schema is sent as a tool the model calls | The model needs the field descriptions to fill values correctly | The model can misname the tool → OutputParserException (step 16) |
| json_mode | "Reply in JSON" with no schema | Flexible shapes | No guarantee which fields come back |
| Prompt + parser (PydanticOutputParser) | Format rules in the prompt, text parsed afterwards | Models without tool or format support | The most fragile: any extra text breaks parsing |

## This step's code

**When to use**

- Code must act on the model's output: if/else, routing, writing queries, scoring chunks.

**Limitations**

- It guarantees the format, not the correctness: valid objects can hold wrong values.
- method="json_schema" never shows the model the field descriptions.
- method="function_calling" can crash when the model misnames the tool (step 16).
- Field descriptions are requests, not rules: "3 queries" can come back as 2.

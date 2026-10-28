# Step 2: prompt templates

Notes for [`step02_prompts.py`](step02_prompts.py).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| ✅ ChatPromptTemplate | Messages with {placeholders}, filled at invoke time | Prompts reused with different values; inside chains | Literal braces must be doubled: {{ }} |
| f-strings / str.format | Plain Python string building | One-off scripts | Not a Runnable, no message roles, no reuse |
| ✅ Zero-shot | Instructions only, no examples | Clear tasks the model already understands | The model may guess the format or style wrong |
| Few-shot | Add worked input → output examples (FewShotChatMessagePromptTemplate) | A specific output style, labels or format | Longer prompts (slower); examples can bias answers |
| ✅ Rules in the system message | Instructions separate from the user's text | Behaviour that should hold for every question | Some models follow system messages more weakly than others |
| Rules in the user message | Instructions and question in one message | Models that ignore system prompts | The user's text can override the rules (prompt injection) |
| MessagesPlaceholder | A slot for a whole list of messages | Chat history (step 19) | History grows; it has to be trimmed |

## This step's code

**When to use**

- The same prompt is used many times with different values (every RAG question).
- Keeping prompt text in one place instead of copy-pasting strings.

**Limitations**

- A literal { or } in the template text (e.g. a JSON example) must be written {{ }}, or it's read as a variable.
- A missing variable is only found at invoke() time (KeyError), not when the template is defined.
- Templates only fill in text: no if/else or loops.

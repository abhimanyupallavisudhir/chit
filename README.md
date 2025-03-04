# chit

`chit` is a git-analogous system for managing LLM chat conversations in a Jupyter notebook. Install as `pip install chitgpt`.

The `chit.Chat` class has methods:

- `commit()` for adding new messages (either user or assistant). For creating an assistant message, the message path leading from the root to the current checked-out message is sent to the LLM.
- `branch()` for creating a new branch at the current checked-out message
- `checkout()` for changing the checkout message. 
- `push()` for dumping pushing to a `Remote` (a json file + an html gui visualization)
- `clone()` a classmethod for initializing a `chit.Chat` object from a json file
- sensible indexing and slicing
- `rm()` for removing a branch or commit
- `mv()` for renaming a branch
- `find()` for finding in conversation history
- `log()` for creating simple tree or forum style visualizations of the chat
- `gui()` for creating a (non-interactive) html gui output of the conversation similar to a classic LLM interface

See [example.ipynb](example.ipynb) for some demonstration, as well as [example2.ipynb](example2.ipynb) where we re-clone an earlier chat and play with it, and [example3.ipynb](example3.ipynb) for demonstrations with tool-calling.

## models

Change the model by directly modifying the `model` attribute e.g.

```python
chat.model = "openrouter/anthropic/claude-3.7-sonnet"
```

We use [litellm](https://github.com/BerriAI/litellm) for the LLM completions, so use their model naming conventions (very useful comprehensive list [here](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)).

## images

Vision is supported, including images from the clipboard like so: `chat.commit("Analyze this image.", image_path = '^V')`.

## tool use

Change tools by modifying the `tools` attribute (which is a list of functions), then **calling `recalc_tools()`**, i.e.

```python
chat.tools.append(web_search)
chat._recalc_tools()
```

Here `web_search` should be a Python function with either (1) a `json` attribute in the [OpenAI specification](https://docs.litellm.ai/docs/completion/function_call) or (2) a numpy-style docstring, which lets us automatically calculate the json attribute using `litellm.utils.function_to_dict`.

Tool-calling is not compatible with streaming. If your chat has tools, you can pass `chat.commit(enable_tools=False)` to temporarily disable tools for that AI call and enable streaming.

## alternative input methods

One convenient util is `chit.read()`, which makes it convenient to paste local file contents into your prompt, i.e.

```python
import chit
from chit import read

chat = chit.Chat()

chat.commit(
f"""
Analyze and review the folowing code:

{read("chit/chit.py")}
"""
)
chat.commit()
```

Another useful thing you may want is to type your prompts in a temporary text file (rather than within a jupyter notebook or python file), e.g. to avoid syntax conflicts (like having to escape quotes manually). You can do so by typing your message as `^N`:

```python
import chit

chat = chit.Chat()

chat.commit("^N") # opens up a temporary file in VS Code
chat.commit()
```

## imports

We have a (probably very rudimentary) importer function for Claude exports, used as follows:

```python
import chit
from chit.import_claude import import_claude
chat = import_claude("claude.json")
```

[Here](https://www.reddit.com/r/ClaudeAI/comments/1ciitou/any_good_tools_for_exporting_chats/) is how you get a Claude export (for a particular chat) -- do *not* use the default Claude data dump in account settings (this does not preserve tree structure); instead load the Claude chat with `Chrome Dev Tools > Network` open and find the correct resource.
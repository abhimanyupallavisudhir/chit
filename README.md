# chit

`chit` is a git-analogous system for managing LLM chat conversations in a Jupyter notebook. Install as `pip install chitgpt`.

The `chit.Chat` class has methods:

- `commit()` for adding new messages (either user or assistant). For creating an assistant message, the message path leading up to the current checked-out message is sent to the LLM.
- `branch()` for creating a new branch at the current checked-out message
- `checkout()` for changing the checkout message. 
- `push()` for dumping pushing to a `remote` (a json file + an html gui visualization)
- `clone()` a classmethod for initializing a `chit.Chat` object from a json file
- sensible indexing and slicing
- `rm()` for removing a branch or commit
- `mv()` for renaming a branch
- `find()` for finding in conversation history
- `log()` for creating simple tree or forum style visualizations of the chat
- `gui()` for creating a (non-interactive) html gui output of the conversation similar to a classic LLM interface

We use [litellm](https://github.com/BerriAI/litellm) for the LLM completions, so use their model naming conventions. Vision is supported, including images from the clipboard like so: `chat.commit("Analyze this image.", image_path = '^V', role="user")`.

See [example.ipynb](example.ipynb) for some demonstration, as well as [example2.ipynb](example2.ipynb) where we re-clone an earlier chat and play with it.

## setting stuff

Change the model by directly modifying the `model` attribute e.g.

```python
chat.model = "claude-3-5-sonnet"
```

We use [litellm](https://github.com/BerriAI/litellm) for the LLM completions, so use their model naming conventions.

Change tools by modifying the `tools` attribute (which is a list of functions), then **calling `recalc_tools()`**, i.e.

```python
chat.tools.append(web_search)
chat._recalc_tools()
```

Here `web_search` should be a function with either (1) a `json` attribute in the [OpenAI specification](https://docs.litellm.ai/docs/completion/function_call) or (2) a numpy-style docstring, which lets us automatically calculate the json attribute using `litellm.utils.function_to_dict`.
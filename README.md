# chit

`chit` is a git-analogous system for managing LLM chat conversations in a Jupyter notebook. Install as `pip install chitgpt`.

The `chit.Chat` class has methods:

- `commit()` for adding new messages (either user or assistant). For creating an assistant message, the message path leading from the root to the current checked-out message is sent to the LLM.
- `branch()` for creating a new branch at the current checked-out message
- `checkout()` for changing the checkout message. 
- `push()` for dumping to a `Remote` (a json file + an html gui visualization) -- note that this will *not* preserve chat settings like the list of tools.
- `clone()` a classmethod for initializing a `chit.Chat` object from a json file
- sensible indexing and slicing
- `rm()` for removing a branch or commit
- `mv()` for renaming a branch
- `show()` for showing a particular message (by commit ID, or any form of indexing)
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

Change tools by modifying the `tools` attribute (which is a list of functions):

```python
chat.tools.append(web_search)
```

Here `web_search` should be a Python function with either (1) a `json` attribute in the [OpenAI specification](https://docs.litellm.ai/docs/completion/function_call) or (2) a numpy-style docstring, which lets us automatically calculate the json attribute using `litellm.utils.function_to_dict`.

Tool-calling is not compatible with streaming. If your chat has tools, you can pass `chat.commit(enable_tools=False)` to temporarily disable tools for that AI call and enable streaming.

## including files

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

## alternate input methods

One useful thing you might like to do, e.g. to avoid syntax conflicts (like having to escape quotes manually), is enter your prompt in a markdown cell, then commit it. To do so first create a markdown cell:

```markdown
/some_name
Enter your prompt here lorem ipsum whatever
```

`some_name` is a variable name for later reference -- just make sure that (1) it begins with a slash, to tell `chit` that it is a prompt, and (2) that it is unique.

Then commit as follows:

```python
chat.commit("^J/some_name")
chat.commit()
```

Using this feature requires that:

1. you have set `chit.config.JUPYTERNB` to the path of your current jupyter notebook. 
2. your notebook is *saved*

(because this feature works by reading and loading the json contents of your jupyter notebook, and [there is no reliable way to get the current jupyter notebook's path](https://stackoverflow.com/questions/52119454/how-to-obtain-jupyter-notebooks-path).)

The above is the *recommended* way to pass complex text prompts. Another option is use Jupyter's built-in text areas to input your text:

```python
import chit
from chit import textput

chat = chit.Chat()

prompt = textput()

# ... write your prompt in the text area that appears, 
# no need to worry about clicking any submit buttons,
# it's dynamic ...
# 
# ... then in a new cell ...

chat.commit(prompt.value)
chat.commit()
```

A more traditional way to do this is to type your prompts in a temporary text file (rather than within a jupyter notebook or python file). We do support this; you can do so by typing your message as `^N`:

```python
import chit

chat = chit.Chat()

chat.commit("^N") # opens up a temporary file in VS Code
chat.commit()
```

And can also specify a specific editor like `^N/code` (which runs `code /tmp/whatever.txt`) or ``^N/gnome-terminal$vim` (which runs `gnome-terminal -- vim /tmp/whatever.txt`). However `^N` is currently quite broken and unreliable, especially in a Jupyter notebook -- it does not work with the `terminal$editor` setup at all, and is unreliable even with `code`. I recommend just using jupyter text areas for now.


## indexing

`chit.Chat` objects support indexing (and slicing) by:
- commit IDs, i.e. `chat['1adca2f6']`
- negative integers, i.e. where `chat[-1]` is the currently checked-out message and you traverse backward from there
- lists, i.e. `chat[["master", "master"]]` to do forward traversal from the currently checked-out message, specifying the branch to choose at each step
- nonnegative integers, i.e. where `chat[0]` is the system message and you traverse forward along the master branch

This also applies to e.g. `rm()`.

## settings

Apart from the `chit.Chat` initialization arguments, we have:

```python
DEFAULT_MODEL
# default: "openrouter/anthropic/claude-3.5-sonnet"
# default model to initialize chit.Chat objects with

chit.config.VERBOSE
# default: True
# enables informational print statements from chit apart from chat responses
# (e.g. telling you how many tool calls are expected to be calculated)

chit.config.FORCE
# default: False
# disables asking for confirmation before removing commits and branches

chit.config.AUTOSAVE
# default: True
# automatically pushes to the Remote, if one is set, after every commit or other change

chit.config.EDITOR
# default: "code"
# default text editor to use for user input if user message is `^N` with no further suffix:
#     `editor-name` for gui editors, e.g. `^N/code`.
#     `terminal-name$editor-name` for terminal editors, e.g. `^N/gnome-terminal$vim`.
#     `$jupyter` to take input from a text area in the Jupyter notebook, i.e. `^N/$jupyter`.
# can be overriden in commit by giving 

chit.config.DISPLAY_CONFIG
# default: {
#     "title": "chitChat",
#     "author": "some worthless pencil pusher",
#     "show_model": True,
#     "show_tools": True,
#     "max_tools": 5,
#     "dark": True,
#     "css": ""
# }
# Configuration for gui visualization. Can be updated with the Chat-specific `display_config` attribute.

chit.config.PRIORITIZE_DATA_REMOTE
# default: False
# whether the remote path stored in the data has priority compared to the path you're actually cloning from.
# Set to False if e.g. you are cloning from a copy or move of the file in a different folder or machine.

chit.config.JUPYTERNB
# default: None
# Set to path of the Jupyter notebook file you're using, in order to be able to use "^J" input.
```

This is the first cell of my own personal chit notebook:

```python
import chit
import chit.config
from chit.utils import read, textput

%load_ext autoreload
%autoreload 2

ALIASES = {
    "claude": "openrouter/anthropic/claude-3.5-sonnet",
    "claude3.7": "openrouter/anthropic/claude-3.7-sonnet",
    "deepseek": "openrouter/deepseek/deepseek-chat",
    "deepseek-r1": "openrouter/deepseek/deepseek-r1",
    "gpt-4o": "openrouter/openai/gpt-4o",
    "gpt-4o-mini": "openrouter/openai/gpt-4o-mini",
    "o1-mini": "openrouter/openai/o1-mini",
    "o1-preview": "openrouter/openai/o1-preview",
    "o1": "openrouter/openai/o1",
    "o3-mini-high": "openrouter/openai/o3-mini-high",
    "o3-mini": "openrouter/openai/o3-mini",
}

chit.config.DEFAULT_MODEL = ALIASES["claude"]
chit.config.VERBOSE = True
chit.config.FORCE = True
chit.config.AUTOSAVE = True
chit.config.EDITOR = "code"
chit.config.PRIORITIZE_DATA_REMOTE=False
chit.config.JUPYTERNB = "/home/manyu/gdrive/Gittable/chithub/doer.ipynb"
chit.config.DISPLAY_CONFIG = {
    "title": "chitChat",
    "author": "Abhimanyu Pallavi Sudhir",
    "show_model": True,
    "show_tools": True,
    "max_tools": 5,
    "dark": True,
    "css": ""
}
```

## imports

We have a (probably very rudimentary) importer function for Claude exports, used as follows:

```python
import chit
chat = chit.Chat.migrate("claude_export.json", format="claude")
```

[Here](https://www.reddit.com/r/ClaudeAI/comments/1ciitou/any_good_tools_for_exporting_chats/) is how you get a Claude export (for a particular chat) -- do *not* use the default Claude data dump in account settings (this does not preserve tree structure); instead load the Claude chat with `Chrome Dev Tools > Network` open and find the correct resource.

## TODO

- [x] improve `^N` input
- [x] make pushing and cloning preserve as much as possible
- [x] autosave feature
- [x] fix html visualization issue
- [x] cleanup this repo
- [x] bugfixes
    - [x] fix visualization bug in global dropdown [PRIORITY]
    - [x] fix claude imports
    - [x] fix "dictionary size changed during iteration" issue when removing commit with a child and a blank child
- [x] Implement better way to do Jupyter notebook inputs based on: [[1]](https://stackoverflow.com/questions/71235359/jupyter-notebook-move-cells-from-one-notebook-into-a-new-notebook/71244733#71244733), [[2]](https://stackoverflow.com/questions/46334525/how-to-copy-multiple-input-cells-in-jupyter-notebook/78123424#78123424) -- maybe the user should preface a markdown block with some name, and we automatically maintain a dict of such names to their following texts, and then send the prompt as `chat.commit('^J/name')` or something. [PRIORITY]
- [ ] html gui improvements
    - [ ] i3-like gui
    - [ ] forum-like gui
    - [x] global dropdown to select a branch; they should be shown with indentations reflecting their nesting
- [ ] add imports from chatgpt, deepseek, xai, gemini, various LLM frontends etc. (only claude currently implemented)

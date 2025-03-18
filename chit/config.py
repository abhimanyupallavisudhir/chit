DEFAULT_MODEL = "openrouter/anthropic/claude-3.5-sonnet"
"""
default: "openrouter/anthropic/claude-3.5-sonnet"
default model to initialize chit.Chat objects with
"""

VERBOSE = True
"""
default: True
enables informational print statements from chit apart from chat responses
(e.g. telling you how many tool calls are expected to be calculated)
Strongly recommend that this be kept to True.
"""

FORCE = False
"""
default: False
disables asking for confirmation before removing commits and branches
"""

AUTOSAVE = True
"""
default: True
automatically pushes to the Remote, if one is set, after every commit or other change
"""

PRIORITIZE_DATA_REMOTE = False
"""
default: False
whether the remote path stored in the data has priority compared to the path you're actually cloning from.
Set to False if e.g. you are cloning from a copy or move of the file in a different folder or machine.
"""

EDITOR = "code"
"""
default text editor to use for user input if user message is `^N` with no further suffix:
    `editor-name` for gui editors, e.g. `^N/code`.
    `terminal-name$editor-name` for terminal editors, e.g. `^N/gnome-terminal$vim`.
    `$jupyter` to take input from a text area in the Jupyter notebook, i.e. `^N/$jupyter`.
"""

DISPLAY_CONFIG = {
    "title": "chitChat",
    "author": "some worthless pencil pusher",
    "show_model": True,
    "show_tools": True,
    "max_tools": 5,
    "dark": True,
    "css": ""
}
"""
Configuration for gui visualization. Can be updated with the Chat-specific `display_config` attribute.
"""

JUPYTERNB = None
"""
default: None
Set to path of the Jupyter notebook file you're using, in order to be able to use "^J" input.
"""
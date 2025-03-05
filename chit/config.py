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

EDITOR = "code"
"""
default text editor to use for user input if user message is `^N` with no further suffix:
    `editor-name` for gui editors, e.g. `^N/code`.
    `terminal-name$editor-name` for terminal editors, e.g. `^N/gnome-terminal$vim`.
    `$jupyter` to take input from a text area in the Jupyter notebook, i.e. `^N/$jupyter`.
"""

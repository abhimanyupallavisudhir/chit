import os
import tempfile
import webbrowser
import warnings
from dataclasses import dataclass
from typing import Optional, Pattern, Any, Literal
from pathlib import Path
import json
import re
import string
import random
import litellm
from litellm import completion, stream_chunk_builder
from chit.utils import wordcel, annoy
from chit.images import prepare_image_message
import chit.config
from litellm.types.utils import (
    # ModelResponse,
    ChatCompletionMessageToolCall,
    Function,
)
from litellm.types.utils import Message as ChatCompletionMessage


@dataclass
class ChitMessage:
    id: str
    message: dict[str, str] | ChatCompletionMessage
    children: dict[str, Optional[str]]  # branch_name -> child_id
    parent_id: Optional[str]
    home_branch: str
    tool_calls: list[ChatCompletionMessageToolCall] | None = None

    @property
    def heir_id(self):
        return self.children[self.home_branch]

    def asdict(self):
        return {
            "id": self.id,
            "message": self.message
            if isinstance(self.message, dict)
            else self.message.json(),
            "children": self.children,
            "parent_id": self.parent_id,
            "home_branch": self.home_branch,
            "tool_calls": [
                tool_call.to_dict()
                if isinstance(tool_call, ChatCompletionMessageToolCall)
                else tool_call
                for tool_call in self.tool_calls
            ]
            if self.tool_calls is not None
            else None,
        }


class Remote:
    def __init__(self, json_file: str | None = None, html_file: str | None = None):
        """
        Initialize a chit.Remote object. This object is used to specify where to save the chat history.

        Arguments:
            json_file (str): either:
                path/to/file.json (str)
                path/to/file (str), interpreted as Remote("path/to/file.json", "path/to/file.html")
            html_file (str): path to the html file to save the chat history to. If json_file does not
                end with .json, this should be left blank to automatically infer it
        """
        if json_file is None and html_file is None:
            raise ValueError("At least one of json_file or html_file must be specified")
        if html_file:
            assert json_file.endswith(".json"), (
                f"Attempted to initialize invalid remote: Remote({json_file}, {html_file})"
            )
            self.json_file = json_file
            self.html_file = html_file
        else:
            if not json_file.endswith(".json"):
                # interpret as creating both json and html
                _json_file = json_file
                json_file = json_file + ".json"
                html_file = _json_file + ".html"
            print(f"Initializing Remote({json_file}, {html_file})")
            self.json_file = json_file
            self.html_file = html_file

    def __str__(self):
        return f"Remote({self.json_file}, {self.html_file})"

    def __repr__(self):
        return f"Remote({self.json_file}, {self.html_file})"


class Chat:
    def __init__(
        self,
        model: str = None,
        tools: list[callable] | None = None,
        remote: str | Remote | None = None,
        display_config: dict | None = None,
    ):
        """
        Initialize a chit.Chat. Any of the below attributes can be set normally later, e.g. `chat.remote = ...`.

        Arguments:
            model (str): model name, in the [LiteLLM specification](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)
                Defaults to chit.config.DEFAULT_MODEL
            tools (list[callable]): list of tools available to the assistant. NOTE:
                - if you use this, streaming will be turned off. You can still pass `enable_tools=False` to `commit()` to disable tools for a single commit.
                - each tool should be a function which either has a `json` attribute of type dict, or has a numpydoc docstring.
            remote (str or Remote): path to a json file to save the chat history to, or a chit.Remote object with json_file and html_file attributes.
                attribute will automatically be calculated from the remote argument passed; see Remote.__init__ for more details.
            display_config (dict): configuration for GUI visualization, e.g.

            display_config={
                "title": "My Chat",
                "author": "Me",
                "favicon": "path/to/favicon.ico",
                "show_model": True,
                "show_tools": True,
                "max_tools": 3,
                "css": \"\"\"
                    .message { border: 2px solid #ddd; }
                    .footer { background: #f0f0f0; }
                \"\"\"
            }

        """
        self.model = model or chit.config.DEFAULT_MODEL
        self.remote: Remote | None = remote
        initial_id = self._generate_short_id()
        self.root_id = initial_id  # Store the root message ID
        self.display_config = chit.config.DISPLAY_CONFIG | (display_config or {})

        # Initialize with system message
        self.messages: dict[str, ChitMessage] = {
            initial_id: ChitMessage(
                id=initial_id,
                message={"role": "system", "content": "You are a helpful assistant."},
                children={"master": None},
                parent_id=None,
                home_branch="master",
            )
        }

        self.current_id = initial_id
        self.current_branch = "master"
        # Track latest message for each branch
        # maps each branch name to the latest message that includes
        # that branch in its children attribute's keys
        self.branch_tips: dict[str, str] = {"master": initial_id}

        self.tools: list[callable] | None = tools

    @property
    def tools(self) -> list[callable] | None:
        # not to be confused with self.tools_, which is a list of tool jsons
        return self._tools

    @tools.setter
    def tools(self, value):
        self._tools = value
        self._recalc_tools()

    @property
    def remote(self) -> Remote | None:
        return self._remote

    @remote.setter
    def remote(self, value):
        if isinstance(value, str):
            value = Remote(value)
        self._remote = value

    def backup(self):
        """
        Save the chat history to the remote, if one is set and autosave is enabled.
        """
        if chit.config.AUTOSAVE and self.remote is not None:
            self.push()

    def _recalc_tools(self):
        if self.tools is None:
            self.tools = []
        for tool in self.tools:
            if not callable(tool):
                raise ValueError("1) what")
            if not hasattr(tool, "json") or not isinstance(tool.json, dict):
                # a tool is a function with an attribute json of type dict.
                # can automatically calculate the json if it has a numpydoc
                # docstring
                json_spec: dict = litellm.utils.function_to_dict(tool)
                tool.json = {"type": "function", "function": json_spec}
        self.tools_ = [tool.json for tool in self.tools]
        self.tool_map = {tool.json["function"]["name"]: tool for tool in self.tools}

    def _generate_short_id(self, length: int = 8) -> str:
        """Generate a short, unique ID of specified length"""
        while True:
            # Create a random string of hexadecimal characters
            new_id = "".join(random.choices(string.hexdigits.lower(), k=length))

            # Ensure it doesn't already exist in our messages
            if not hasattr(self, "messages") or new_id not in self.messages:
                return new_id

    def _generate_new_branch_name(self, branch_name: str) -> str:
        """Generate a new branch name based on the current branch name"""
        import re

        new_branch_name = branch_name
        while new_branch_name in self.branch_tips:
            match = re.match(r"^(.+)_(\d+)$", new_branch_name)
            if match:
                new_branch_name = f"{match.group(1)}_{int(match.group(2)) + 1}"
            else:
                new_branch_name = f"{new_branch_name}_1"
        return new_branch_name

    def commit(
        self,
        message: str | None = None,
        image_path: str | Path | None = None,
        role: str = None,
        enable_tools=True,
    ) -> str:
        """
        Commit a message to the chat history.

        Arguments:
            message (str): message content
            image_path (str or Path): path to an image to include in the message
            role (str): role of the message (user, assistant, system, tool)
            enable_tools (bool): turn off to disable tool use in a chat that
                otherwise has tools (e.g. to enable streaming)
        """
        if message and message.startswith("^N"):
            # Parse editor specification
            editor_spec = message[2:].strip("/ ")
            message = self._capture_editor_content(editor_spec)
        if message and message.startswith("^J"):
            variable_spec = message[2:].strip("/ ")
            message = self.jupyter_inputs[
                variable_spec
            ]  # let it raise an error if not present
        if role is None:  # automatically infer role based on current message
            current_role = self[self.current_id].message["role"]
            if current_role == "system":
                role = "user"
            elif current_role == "user":
                role = "assistant"
            elif current_role == "assistant":
                if self[self.current_id].tool_calls:
                    role = "tool"
                else:
                    role = "user"
            elif current_role == "tool":
                if self[self.current_id].tool_calls:
                    role = "tool"
                else:
                    role = "assistant"
            else:
                raise ValueError(f"current_role {current_role} not supported")
        # allow short roles
        ROLE_SHORTS = {"u": "user", "a": "assistant", "s": "system"}
        role = ROLE_SHORTS.get(role.lower(), role)
        existing_child_id = self.messages[self.current_id].children[self.current_branch]

        # check that checked-out message does not already have a child in the checked-out branch
        if existing_child_id is not None:
            new_branch_name = self._generate_new_branch_name(self.current_branch)
            wordcel(
                f"WARNING: Current message {self.current_id} already has a child message {existing_child_id} on branch {self.current_branch}. "
                f"Creating new branch {new_branch_name} to avoid overwriting."
            )
            self.branch(new_branch_name, checkout=True)

        new_id = self._generate_short_id()

        if image_path is not None:
            assert role == "user", "Only user messages can include images"
            message = prepare_image_message(message, image_path)

        response_tool_calls = None  # None by default unless assistant calls for it or we have some from previous tool call

        if role == "user":
            assert message is not None or image_path is not None, (
                "User message cannot be blank"
            )
            message_full = {"role": role, "content": message}

        if role == "assistant" and message is not None:
            # put words in mouth
            message_full = {"role": role, "content": message}

        if role == "assistant" and message is None:
            # Generate AI response
            history = self._get_message_history()
            if hasattr(self, "tools_") and self.tools_ and enable_tools:
                response = completion(
                    model=self.model,
                    messages=history,
                    tools=self.tools_,
                    tool_choice="auto",
                    stream=False,
                )
                message_full: ChatCompletionMessage = response.choices[0].message
                print(message_full.content)
                response_tool_calls: list[ChatCompletionMessageToolCall] | None = (
                    message_full.tool_calls
                )
            else:
                _response = completion(model=self.model, messages=history, stream=True)
                chunks = []
                for chunk in _response:
                    print(chunk.choices[0].delta.content or "", end="")
                    chunks.append(chunk)
                response = stream_chunk_builder(chunks, messages=history)
                message_full: ChatCompletionMessage = response.choices[0].message

        if role == "tool":
            # when we pop tool calls, it should not modify previous history
            response_tool_calls = self.current_message.tool_calls.copy()
            if not response_tool_calls:
                raise ValueError("No tool calls requested to call")
            t: ChatCompletionMessageToolCall = response_tool_calls.pop(0)
            f: Function = t.function
            f_name: str = f.name
            f_args: str = f.arguments
            if f_name not in self.tool_map:
                warnings.warn(f"Tool {f_name} not found in tool_map; skipping")
                tool_result = f"ERROR: Tool {f_name} not found"
            else:
                tool: callable = self.tool_map[f_name]
                tool_kwargs: dict = json.loads(f_args)
                try:
                    tool_result: Any = tool(**tool_kwargs)
                except Exception as e:
                    tool_result: str = f"ERROR: {e}"
                message = str(tool_result)
                message_full = {
                    "role": "tool",
                    "content": message,
                    "tool_call_id": t.id,
                    "name": f_name,
                }

        # Create new message
        new_message = ChitMessage(
            id=new_id,
            message=message_full,
            tool_calls=response_tool_calls,
            children={self.current_branch: None},
            parent_id=self.current_id,
            home_branch=self.current_branch,
        )

        # Update parent's children
        self.messages[self.current_id].children[self.current_branch] = new_id

        # Add to messages dict
        self.messages[new_id] = new_message

        # Update branch tip
        self.branch_tips[self.current_branch] = new_id

        # Update checkout
        self.current_id = new_id

        if response_tool_calls:
            wordcel(
                f"<<<{len(response_tool_calls)} tool calls pending; "
                f"use .commit() to call one-by-one>>>"
            )

        self.backup()

        # return new_message.message["content"]

    def _capture_editor_content(self, editor_spec=None):
        """
        Open editor and capture content based on editor specification.

        Editor spec formats:
        - None: use self.editor default
        - "code", "gnome-text-editor", etc: GUI editor
        - "gnome-terminal$vim", "xterm$nano": terminal$editor
        - "$jupyter": use Jupyter text area
        """
        if not editor_spec:
            editor_spec = chit.config.EDITOR

        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            temp_path = f.name

        if "$" in editor_spec:
            # Terminal-based editor
            terminal, editor = editor_spec.split("$")
            os.system(f"{terminal} -- {editor} {temp_path}")
        else:
            # GUI editor
            os.system(f"{editor_spec} {temp_path}")

        input("Press Enter when you're done editing... ")

        with open(temp_path, "r") as f:
            content = f.read()

        os.unlink(temp_path)

        if content.strip():
            return content
        else:
            raise ValueError("No content added to editor")

    @property
    def jupyter_inputs(self) -> dict:
        import nbformat as nbf

        ntbk: dict = nbf.read(chit.config.JUPYTERNB, nbf.NO_CONVERT)
        cells: list[dict] = ntbk["cells"]
        md_cells: list[dict] = [
            cell for cell in cells if cell["cell_type"] == "markdown"
        ]
        cell_map: dict = {}  # dict of first line of cell content to cell
        # first line as in stuff between / and \n
        for cell in md_cells:
            text: str = cell["source"]
            if not text.startswith("/"):
                continue
            lines: list[str] = text.split("\n")
            first_line: str = lines[0].strip("/ ")
            contents: str = "\n".join(lines[1:])
            if first_line in cell_map:
                warnings.warn(
                    f"Prompt name {first_line} associated with multiple cells "
                    f"{cell_map[first_line][:30]} and {contents[:30]}; using the latter"
                )
            cell_map[first_line] = contents
        return cell_map

    def branch(self, branch_name: str, checkout: bool = True) -> None:
        if branch_name in self.branch_tips:
            raise ValueError(
                f"Branch '{branch_name}' already exists (latest at message {self.branch_tips[branch_name]})"
            )

        self.messages[self.current_id].children[branch_name] = None
        self.branch_tips[branch_name] = self.current_id
        if checkout:
            old_id = self.current_id
            self.checkout(branch_name=branch_name)
            assert (
                self.current_id == old_id
            )  # since we just created the branch, it should be the same as before

        self.backup()

    def show(
        self, message_id: str, mode: Literal["print", "return"] = "print"
    ) -> None | str:
        """Print the content of a message.

        Arguments:
            message_id (str): ID of the message to show
            mode (str): whether to print the content or return it.
        """
        content = self[message_id].message["content"]
        if mode == "print":
            print(content)
        elif mode == "return":
            return content
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _resolve_forward_path(
        self, branch_path: list[str], start_id: Optional[str] = None
    ) -> str:
        """Follow a path of branches forward from start_id (or current_id if None)"""
        current = start_id if start_id is not None else self.current_id

        for branch in branch_path:
            current_msg = self.messages[current]
            if branch not in current_msg.children:
                raise KeyError(f"Branch '{branch}' not found in message {current}")

            next_id = current_msg.children[branch]
            if next_id is None:
                raise IndexError(
                    f"Branch '{branch}' exists but has no message in {current}"
                )

            current = next_id

        return current

    def _resolve_negative_index(self, index: int) -> str:
        """Convert negative index to message ID by walking up the tree"""
        if index >= 0:
            raise ValueError("This method only handles negative indices")

        current = self.current_id
        steps = -index - 1  # -1 -> 0 steps, -2 -> 1 step, etc.

        for _ in range(steps):
            current_msg = self.messages[current]
            if current_msg.parent_id is None:
                raise IndexError("Chat history is not deep enough")
            current = current_msg.parent_id

        return current

    def _resolve_nonnegative_index(self, index: int) -> str:
        """Convert positive or zero index to message ID by following master branch from root"""
        if index < 0:
            raise ValueError("This method only handles non-negative indices")

        current = self.root_id
        steps = index  # 0 -> root, 1 -> first message, etc.

        for _ in range(steps):
            current_msg = self.messages[current]
            if "master" not in current_msg.children:
                raise IndexError("Chat history not long enough (no master branch)")
            next_id = current_msg.children["master"]
            if next_id is None:
                raise IndexError("Chat history not long enough (branch ends)")
            current = next_id

        return current

    def checkout(
        self,
        message_id: Optional[str | int | list[str]] = None,
        branch_name: Optional[str] = None,
    ) -> None:
        """Checkout to a specific message or branch

        Arguments:
            message_id (str, int, list[str]): ID of the message to checkout to. Can be:
                - message ID (str)
                - non-negative index (int) to count back from the root
                - negative index (int) to count back from the current message
                - list of branch names to follow forward from the current message
            branch_name (str): name of the branch to checkout to
        """
        if message_id is not None:
            if isinstance(message_id, int):
                if message_id >= 0:
                    message_id = self._resolve_nonnegative_index(message_id)
                else:
                    message_id = self._resolve_negative_index(message_id)
            elif isinstance(message_id, list):
                if not all(isinstance(k, str) for k in message_id):
                    raise TypeError("Branch path must be a list of strings")
                message_id = self._resolve_forward_path(message_id)
            elif message_id not in self.messages:
                raise ValueError(f"Message {message_id} does not exist")
            self.current_id = message_id

        if branch_name is not None:
            if branch_name not in self.branch_tips:
                raise ValueError(f"Branch '{branch_name}' does not exist")
            # Always checkout to the latest message containing this branch
            if message_id is None:
                self.current_id = self.branch_tips[branch_name]
            else:
                assert branch_name in self.messages[message_id].children, (
                    f"Branch {branch_name} not found in message {message_id}"
                )
            self.current_branch = branch_name
        else:
            self.current_branch = self.messages[self.current_id].home_branch

        self.backup()

    def _get_message_history(self) -> list[dict[str, str]]:
        """Reconstruct message history from current point back to root"""
        history = []
        current = self.current_id

        while current is not None:
            msg = self.messages[current]
            history.insert(0, msg.message)
            current = msg.parent_id

        return history

    def asdict(self):
        return {
            "model": self.model,
            "tools_": self.tools_,
            "remote": vars(self.remote) if self.remote is not None else None,
            "display_config": self.display_config
            if self.display_config is not None
            else None,
            "messages": {k: v.asdict() for k, v in self.messages.items()},
            "current_id": self.current_id,
            "current_branch": self.current_branch,
            "root_id": self.root_id,
            "branch_tips": self.branch_tips,
        }

    def push(self) -> None:
        """Save chat history to configured remote"""
        if self.remote is None:
            raise ValueError("No remote configured. Set chat.remote first.")

        if self.remote.json_file is not None:
            data = self.asdict()
            os.makedirs(os.path.dirname(self.remote.json_file), exist_ok=True)
            with open(self.remote.json_file, "w") as f:
                json.dump(data, f)

        if self.remote.html_file is not None:
            html_content = self._generate_viz_html()
            os.makedirs(os.path.dirname(self.remote.html_file), exist_ok=True)
            with open(self.remote.html_file, "w") as f:
                f.write(html_content)

    def __getitem__(
        self, key: str | int | list[str] | slice
    ) -> ChitMessage | list[ChitMessage]:
        # Handle string indices (commit IDs)
        if isinstance(key, str):
            if key not in self.messages:
                raise KeyError(f"Message {key} does not exist")
            return self.messages[key]

        # Handle integer indices
        if isinstance(key, int):
            if key >= 0:
                return self.messages[self._resolve_nonnegative_index(key)]
            return self.messages[self._resolve_negative_index(key)]

        # Handle forward traversal via branch path
        if isinstance(key, list):
            if not all(isinstance(k, str) for k in key):
                raise TypeError("Branch path must be a list of strings")
            return self.messages[self._resolve_forward_path(key)]

        # Handle slices
        if isinstance(key, slice):
            if key.step is not None:
                raise ValueError("Step is not supported in slicing")

            # Convert start/stop to message IDs
            start_id = None
            if isinstance(key.start, int):
                if key.start >= 0:
                    start_id = self._resolve_nonnegative_index(key.start)
                else:
                    start_id = self._resolve_negative_index(key.start)
            elif isinstance(key.start, list):
                start_id = self._resolve_forward_path(key.start)
            else:
                start_id = key.start

            stop_id = None
            if isinstance(key.stop, int):
                if key.stop >= 0:
                    stop_id = self._resolve_nonnegative_index(key.stop)
                else:
                    stop_id = self._resolve_negative_index(key.stop)
            elif isinstance(key.stop, list):
                stop_id = self._resolve_forward_path(key.stop)
            else:
                stop_id = key.stop

            # Walk up from stop_id to start_id
            result = []
            current = stop_id if stop_id is not None else self.current_id

            while True:
                if current is None:
                    raise IndexError("Reached root before finding start")

                result.append(self.messages[current])

                if current == start_id:
                    break

                current = self.messages[current].parent_id

            return result[::-1]  # Reverse to get chronological order

        raise TypeError(f"Invalid key type: {type(key)}")

    @classmethod
    def clone(
        cls,
        remote: str | Remote,
        prioritize_data_remote: bool = chit.config.PRIORITIZE_DATA_REMOTE,
    ) -> "Chat":
        """Create new Chat instance from remote file

        Arguments:
            remote (str or Remote): where to load chat history from. Can be:
                    - /path/to/file.json (str)
                    - /path/to/file (str), interpreted as Remote("/path/to/file.json", "/path/to/file.html")
                    - Remote("/path/to/file.json", "/path/to/file.html")
                    - ("/path/to/file.json", "/path/to/file.html") (tuple) -- interpreted as Remote object
                or a chit.Remote object with json_file and html_file attributes.
                The one situation it may make sense to have it be a chit.Remote is to
                specify the full remote of the cloned object, including the html file
                -- make sure to use prioritize_data_remote = False if you want to do this!
            prioritize_data_remote (bool): whether the remote path stored in the data has
                priority compared to the path you're actually cloning from. Set to
                False if e.g. you are cloning from a copy or move of the file in a
                different folder or machine.

        """
        if isinstance(remote, tuple):
            remote = Remote(*remote)
        if isinstance(remote, Remote):
            remote_str: str = remote.json_file
            remote_dict: dict = vars(remote)
        elif isinstance(remote, str):
            if remote.endswith(".json"):
                remote_str: str = remote
                remote_dict: dict = {"json_file": remote_str}
            else:
                # interpret as creating both json and html
                remote_str: str = remote + ".json"
                remote_dict: dict = {
                    "json_file": remote_str,
                    "html_file": remote + ".html",
                }
        else:
            raise ValueError(
                f"unrecognized remote type {type(remote)}; must be str, Remote or tuple"
            )

        with open(remote_str, "r") as f:
            data = json.load(f)

        data_remote_dict = data.get("remote", {})
        if prioritize_data_remote:
            # data_remote has priority
            updated_remote = Remote(**(remote_dict | data_remote_dict))
        else:
            # remote has priority
            updated_remote = Remote(**(data_remote_dict | remote_dict))
        wordcel(f"Remote specified in data: {Remote(**data_remote_dict)}")
        wordcel(f"Remote specified in argument: {Remote(**remote_dict)}")
        wordcel(f"Using remote: {updated_remote}")
        chat = cls(
            model=data.get("model", chit.config.DEFAULT_MODEL),
            tools=None,
            remote=updated_remote,
            display_config=data.get("display_config", chit.config.DISPLAY_CONFIG),
        )

        chat.messages = {k: ChitMessage(**v) for k, v in data["messages"].items()}
        chat.current_id = data["current_id"]
        chat.current_branch = data["current_branch"]
        chat.root_id = data["root_id"]
        chat.branch_tips = data["branch_tips"]
        if data.get("_tools", None):
            wordcel(
                f"WARNING: found the following tools in the remote: {data['_tools']} "
                "but cannot add them as we do not have the functions. Please add them manually."
            )
        return chat

    @property
    def current_message(self):
        return self[self.current_id]

    def _is_descendant(self, child_id: str, ancestor_id: str) -> bool:
        """
        Test if ancestor_id is an ancestor of child_id

        Args:
            child_id: ID of the potential descendant
            ancestor_id: ID of the potential ancestor

        Returns:
            bool: True if ancestor_id is an ancestor of child_id, False otherwise

        Raises:
            ValueError: If either ID doesn't exist in the chat
        """
        if child_id not in self.messages:
            raise ValueError(f"Message {child_id} does not exist")

        if ancestor_id not in self.messages:
            raise ValueError(f"Message {ancestor_id} does not exist")

        # If they're the same, return False (not a true ancestor)
        if child_id == ancestor_id:
            return True

        # Traverse up from child until we either find the ancestor or reach the root
        current = self.messages[child_id].parent_id

        while current is not None:
            if current == ancestor_id:
                return True
            current = self.messages[current].parent_id

        return False

    def _get_branch_root(self, branch_name: str) -> str:
        """
        Find the first commit where a branch was created (the branch root)

        Args:
            branch_name: Name of the branch to find the root for

        Returns:
            str: ID of the branch root message

        Raises:
            ValueError: If the branch doesn't exist
        """
        if branch_name not in self.branch_tips:
            raise ValueError(f"Branch '{branch_name}' does not exist")

        # Start from the branch tip
        current_id = self.branch_tips[branch_name]

        # Walk up the parent chain until we find a message with a different home_branch
        while True:
            current_msg = self.messages[current_id]

            # If this is the root message, it's the root of all branches
            if current_msg.parent_id is None:
                return current_id

            # Get the parent message
            parent_id = current_msg.parent_id
            parent_msg = self.messages[parent_id]

            # If the parent has a different home branch, then current_id is the branch root
            if (
                current_msg.home_branch == branch_name
                and parent_msg.home_branch != branch_name
            ):
                return current_id

            # Move up to the parent
            current_id = parent_id

            # Safety check - if we reach a message without a home_branch, something's wrong
            if not hasattr(current_msg, "home_branch"):
                raise ValueError(
                    f"Invalid message structure: missing home_branch at {current_id}"
                )

    def _check_kalidasa_branch(self, branch_name: str) -> tuple[str, str]:
        """
        Check if we are trying to cut the branch we are checked out on (via an
        ancestral branch), and return the commit and branch we must checkout to to cut it.
        """
        current_id = self.current_id
        current_branch = self.current_branch
        current_message = self[current_id]
        if current_branch == branch_name:
            current_branch = current_message.home_branch
        while True:
            if current_message.home_branch == branch_name:
                current_id = current_message.parent_id
                current_branch = self[current_id].home_branch
                if (
                    current_id is None
                ):  # nothing we can do if you're trying to delete master
                    break
                else:
                    current_message = self[current_id]
            else:
                break
        return current_id, current_branch

    def _check_kalidasa_commit(self, commit_id: str) -> tuple[str, str]:
        """Check if we are trying to cut the branch we are checked out on (via an
        ancestral commit), and return the commit and branch we must checkout to cut it.
        """
        if self._is_descendant(child_id=self.current_id, ancestor_id=commit_id):
            parent_id = self[commit_id].parent_id
            if parent_id is None:
                raise ValueError("Cannot delete root message")
            parent_message = self[parent_id]
            return parent_id, parent_message.home_branch
        else:
            return self.current_id, self.current_branch

    def _rm_branch(self, branch_name: str) -> None:
        """Remove all messages associated with a branch."""
        # Check if we're trying to remove current branch or home branch
        self.checkout(*self._check_kalidasa_branch(branch_name))

        # First pass: identify messages to delete and clean up their parent references
        to_delete = set()
        parent_cleanups = []  # List of (parent_id, msg_id) tuples to clean up

        for msg_id, msg in self.messages.items():
            if msg.home_branch == branch_name:
                to_delete.add(msg_id)
                if msg.parent_id is not None:
                    parent_cleanups.append((msg.parent_id, msg_id))
            if branch_name in msg.children:
                # to_delete.add(msg.children[branch_name]) # no need to do this now
                del msg.children[branch_name]

        # Clean up parent references
        for parent_id, msg_id in parent_cleanups:
            if parent_id in self.messages:  # Check parent still exists
                parent = self.messages[parent_id]
                # Find and remove this message from any branch in parent's children
                for branch, child_id in list(
                    parent.children.items()
                ):  # Create list copy to modify during iteration
                    if child_id == msg_id:
                        del parent.children[branch]

        # Finally delete the messages
        for msg_id in to_delete:
            if msg_id in self.messages:  # Check message still exists
                del self.messages[msg_id]

        # Remove from branch_tips if present
        if branch_name in self.branch_tips:
            del self.branch_tips[branch_name]

    def _rm_commit(self, commit_id: str) -> None:
        """Remove a commit and all its children."""
        if commit_id not in self.messages:
            raise ValueError(f"Message {commit_id} does not exist")

        message = self.messages[commit_id]

        # if removing the current commit or an ancestor, checkout its parent
        self.checkout(*self._check_kalidasa_commit(commit_id))

        # kill all children
        for child_branch, child_id in message.children.items():
            if child_id is not None:
                self._rm_commit(child_id)
            if child_branch != message.home_branch:
                # this is a branch that doesn't exist anywhere else, so it must be removed
                self.branch_tips.pop(child_branch, None)

        # Update parent's children
        if message.parent_id is not None:
            parent = self.messages[message.parent_id]
            for branch, child_id in parent.children.items():
                if child_id == commit_id:
                    parent.children[branch] = None

        for branch, tip_id in self.branch_tips.items():
            if tip_id == commit_id:
                # if parent is also in the same branch, that's the new tip
                if self[message.parent_id].home_branch == branch:
                    self.branch_tips[branch] = message.parent_id
                else:
                    del self.branch_tips[branch]

        # Delete the message
        del self.messages[commit_id]

        return

    def rm(
        self, commit_id: str | int | None = None, branch_name: str | None = None
    ) -> None:
        """
        Remove a commit or branch.

        Args:
            commit_id (str | int | None): ID (or alternate indexing) of the commit to remove.
                If None, branch_name must be specified.
            branch_name (str | None): Name of the branch to remove. If None, commit_id must
                be specified.

        """
        if isinstance(commit_id, int):
            # allow negative and positive indices
            if commit_id > 0:
                commit_id = self._resolve_nonnegative_index(commit_id)
            else:
                commit_id = self._resolve_negative_index(commit_id)
        if not annoy(
            f"Are you sure you want to delete {'commit ' + commit_id if commit_id else 'branch ' + branch_name}?"
        ):
            return
        if commit_id is not None:
            if branch_name is not None:
                raise ValueError(
                    "cannot specify both commit_name and branch_name for rm"
                )
            self._rm_commit(commit_id)
        elif branch_name is not None:
            self._rm_branch(branch_name)
        self.backup()

    def mv(self, branch_name_old: str, branch_name_new: str) -> None:
        """Rename a branch throughout the tree.

        Args:
            branch_name_old (str): Name of the branch to rename
            branch_name_new (str): New name for the branch
        """
        if branch_name_new in self.branch_tips:
            raise ValueError(f"Branch '{branch_name_new}' already exists")

        # Update all references to the branch
        for msg in self.messages.values():
            # Update children dict keys
            if branch_name_old in msg.children:
                msg.children[branch_name_new] = msg.children.pop(branch_name_old)

            # Update home_branch
            if msg.home_branch == branch_name_old:
                msg.home_branch = branch_name_new

        # Update branch_tips
        if branch_name_old in self.branch_tips:
            self.branch_tips[branch_name_new] = self.branch_tips.pop(branch_name_old)

        # Update current_branch if needed
        if self.current_branch == branch_name_old:
            self.current_branch = branch_name_new

        self.backup()

    def find(
        self,
        pattern: str | Pattern,
        *,
        case_sensitive: bool = False,
        roles: Optional[list[str]] = None,
        max_results: Optional[int] = None,
        regex: bool = False,
        context: int = 0,  # Number of messages before/after to include
    ) -> list[dict[str, ChitMessage | list[ChitMessage]]]:
        """
        Search for messages matching the pattern.

        Args:
            pattern: String or compiled regex pattern to search for
            case_sensitive: Whether to perform case-sensitive matching
            roles: List of roles to search in ("user", "assistant", "system"). None means all roles.
            max_results: Maximum number of results to return. None means return all matches.
            regex: Whether to treat pattern as a regex (if string)
            context: Number of messages before/after to include in results

        Returns:
            List of dicts, each containing:
                - 'match': Message that matched
                - 'context': List of context messages (if context > 0)
        """
        if isinstance(pattern, str) and not regex:
            pattern = re.escape(pattern)

        if isinstance(pattern, str):
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(pattern, flags)

        results = []

        # Walk through messages in chronological order from root
        current_id = self.root_id
        message_sequence = []

        while current_id is not None:
            message = self.messages[current_id]
            message_sequence.append(message)

            # Check if message matches search criteria
            if (roles is None or message.message["role"] in roles) and pattern.search(
                message.message["content"]
            ):
                # Get context if requested
                context_messages = []
                if context > 0:
                    start_idx = max(0, len(message_sequence) - context - 1)
                    end_idx = min(
                        len(message_sequence) + context, len(message_sequence)
                    )
                    context_messages = message_sequence[start_idx:end_idx]
                    context_messages.remove(
                        message
                    )  # Don't include the match itself in context

                results.append({"match": message, "context": context_messages})

                if max_results and len(results) >= max_results:
                    break

            # Move to next message on master branch
            current_id = message.children.get("master")

        return results

    def _process_commit_id(self, commit_id: str):
        """Helper function for Chat.log()"""
        commit = self.messages[commit_id]
        commit_id_proc = commit_id
        role = commit.message["role"]
        prefix = f"[{role[0].upper()}{'*' if commit_id == self.current_id else '_'}]"
        commit_id_proc = prefix + commit_id_proc
        return commit_id_proc

    def _process_branch_name(self, branch_name: str):
        """Helper function for Chat.log()"""
        if branch_name == self.current_branch:
            return f" ({branch_name}*)"
        return f" ({branch_name})"

    def _log_tree_draw_from(self, frontier_id: str, branch_name: str) -> list[str]:
        """Helper function for Chat.log()"""
        log_lines: list[str] = []
        log_lines.append(self._process_commit_id(frontier_id))
        frontier: ChitMessage = self.messages[frontier_id]

        horizontal_pos: int = len(log_lines[0])  # position where stuff should be added

        if hasattr(frontier, "heir_id"):
            log_lines[0] += "──"
            if frontier.heir_id is None:
                log_lines[0] += self._process_branch_name(branch_name)
            else:
                subtree: list[str] = self._log_tree_draw_from(
                    frontier.heir_id, frontier.home_branch
                )
                # we would like to just append subtree to the current log
                # but it's actually multiple lines that need to get appended
                # to the right propositions
                indent: int = len(log_lines[0])
                log_lines[0] += subtree[0]
                for subtree_line in subtree[1:]:
                    log_lines.append(" " * indent + subtree_line)

        for child_branch, child_id in frontier.children.items():
            if child_branch == frontier.home_branch:
                # already processed the heir
                continue
            else:
                for i in range(len(log_lines)):
                    if i == 0:
                        continue
                    line = log_lines[i]
                    if line[horizontal_pos] == "└":  # no longer the final branch
                        line = line[:horizontal_pos] + "├" + line[horizontal_pos + 1 :]
                    if line[horizontal_pos] == " ":  # extend
                        line = line[:horizontal_pos] + "│" + line[horizontal_pos + 1 :]
                    log_lines[i] = line
                log_lines.append(" " * horizontal_pos + "└─")
                if child_id is None:
                    log_lines[-1] += self._process_branch_name(child_branch)
                else:
                    subtree: list[str] = self._log_tree_draw_from(
                        child_id, child_branch
                    )
                    indent: int = horizontal_pos + 1  # the length of log_lines[-1]
                    log_lines[-1] += subtree[0]
                    for subtree_line in subtree[1:]:
                        log_lines.append(" " * indent + subtree_line)

        # if not frontier.children or all(v is None for v in frontier.children.values()):
        #     log_lines[0] += self._process_branch_name(branch_name)
        return log_lines

    def _log_tree(self) -> str:
        """
        Generate a tree visualization of the conversation history, like this:

        ```
        001e1e──ab2839──29239b──f2foif9──f2f2f2 (master)
                      ├─bb2b2b──adaf938 (features)
                      |       └─f2f2f2*──aa837r (design_discussion*)
                      |                        ├ (flask_help)
                      |                        └ (tree_viz_help)
                      └─r228df──f2f2f2 (publishing)
                              └─j38392──b16327 (pypi)
        ```
        """
        log_lines: list[str] = self._log_tree_draw_from(self.root_id, "master")
        res = "\n".join(log_lines)
        return res

    def _process_message_content(self, content: str | list[dict[str, str]]) -> str:
        if isinstance(content, list):
            content_proc = "<IMG>"
            for item in content:
                if item["type"] == "text":
                    content_proc += item["text"]
                    break
        else:
            content_proc = content
        content_proc = content_proc.replace("\n", r" ").strip()[:57] + "..."
        return content_proc

    def _log_forum_draw_from(self, frontier_id: str) -> list[str]:
        log_lines: list[str] = []
        frontier: ChitMessage = self[frontier_id]
        log_lines.append(
            f"{self._process_commit_id(frontier_id)}: {self._process_message_content(frontier.message['content'])}"
        )
        # show heir first
        if hasattr(frontier, "heir_id"):
            if frontier.heir_id is None:
                log_lines[0] += self._process_branch_name(frontier.home_branch)
            else:
                subtree: list[str] = self._log_forum_draw_from(
                    frontier.heir_id
                )  # recurse
                subtree_ = [" " * 4 + line for line in subtree]  # indent
                log_lines.extend(subtree_)
        for child_branch, child_id in frontier.children.items():
            if child_branch == frontier.home_branch:
                continue  # already processed the heir
            elif child_id is None:
                log_lines.append(" " * 4 + self._process_branch_name(child_branch))
            else:
                subtree: list[str] = self._log_forum_draw_from(child_id)  # recurse
                subtree_ = [" " * 4 + line for line in subtree]  # indent
                log_lines.extend(subtree_)
        # if not frontier.children or all(v is None for v in frontier.children.values()):
        #     log_lines[0] += self._process_branch_name(frontier.home_branch)
        return log_lines

    def _log_forum(self) -> str:
        """
        Generate a forum-style visualization of the conversation history, like this:

        ```
        [S] 001e1e: You are a helpful assista...
            [U] ab2839: Hello I am Dr James and I...
                [A] 29239b: Hello Dr James, how can I...
                    [U] f2foif9: I am trying to use the...
                        [A] f2f2f2: Have you tried using... (master)
                [A] bb2b2b: Hello Dr James, I see you...
                    [U] adaf938: Can we implement the featur... (features)
                    [U*] f2f2f2: This is actually a design issue...
                        [A] aa837r: Sure, I'll help you design a React... (design_discussion*)
                            (flask_help)
                            (tree_viz_help)
                [A] r228df: I see you are working on the...
                    [U] f2f2f2: Ok that worked. Now let's pu... (publishing)
                    [U] j38392: How do I authenticate with p...
                        [A] b16327: Since you are working with g... (pypi)
        ```
        """
        log_lines: list[str] = self._log_forum_draw_from(self.root_id)
        res = "\n".join(log_lines)
        return res

    def gui(
        self,
        file_path: Optional[str | Path] = None,
        mode: Literal["print", "return"] = "print",
    ) -> None:
        """
        Create and open an interactive visualization of the chat tree.

        Args:
            file_path: Optional path where the HTML file should be saved.
                    If None, creates a temporary file instead.
            mode: Whether to print the HTML content or return it as a string.
        """
        html_content = self._generate_viz_html()

        if mode == "return":
            return html_content
        elif mode == "print":
            if file_path is not None:
                # Convert to Path object if string
                path = Path(file_path)
                # Create parent directories if they don't exist
                path.parent.mkdir(parents=True, exist_ok=True)
                # Write the file
                path.write_text(html_content)
                # Open in browser
                webbrowser.open(f"file://{path.absolute()}")
            else:
                # Original temporary file behavior
                with tempfile.NamedTemporaryFile(
                    "w", suffix=".html", delete=False
                ) as f:
                    f.write(html_content)
                    temp_path = f.name
                webbrowser.open(f"file://{temp_path}")
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _prepare_messages_for_viz(self) -> dict[str, Any]:
        """Convert messages to a format suitable for visualization."""
        return {
            "messages": {
                k: {
                    "id": m.id,
                    "message": m.message
                    if isinstance(m.message, dict)
                    else m.message.json(),
                    "children": m.children,
                    "parent_id": m.parent_id,
                    "home_branch": m.home_branch,
                }
                for k, m in self.messages.items()
            },
            "current_id": self.current_id,
            "current_branch": self.current_branch,
            "root_id": self.root_id,
        }

    def _generate_viz_html(self) -> str:
        """Generate the HTML for visualization."""
        data = self._prepare_messages_for_viz()
        data_str = json.dumps(data).replace("</", "<\\/")

        self.display_config = getattr(
            self, "display_config", chit.config.DISPLAY_CONFIG
        )

        # Get display configuration
        display_title = self.display_config.get("title", "chit conversation")
        author = self.display_config.get("author", "some worthless pencil pusher")
        favicon = self.display_config.get("favicon", "")
        show_model = self.display_config.get("show_model", False)
        show_tools = self.display_config.get("show_tools", False)
        max_tools = self.display_config.get("max_tools", None)
        custom_css = self.display_config.get("css", "")

        # Prepare author info
        if show_model:
            author_info = f"A conversation between {author} and {self.model}. "
        else:
            author_info = f"A conversation between {author} and his AI overlord. "

        # Prepare tools info
        tools_info = ""
        if show_tools and self.tools_:
            tool_list = self.tools_[:max_tools] if max_tools else self.tools_
            tools_str = ", ".join(f"`{t['function']['name']}`" for t in tool_list)
            if max_tools and len(self.tools_) > max_tools:
                tools_str += f" and {len(self.tools_) - max_tools} more"
            tools_info = f"Available tools: {tools_str}"

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{display_title} | chit</title>
    <meta charset="UTF-8">
    {'<link rel="icon" href="' + favicon + '">' if favicon else ""}
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js"></script>
    <style>
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px 20px 60px 20px;  /* Added padding for footer */
            background: #f5f5f5;
        }}
        .message {{
            margin: 20px 0;
            padding: 15px;
            border-radius: 10px;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            position: relative;  /* For absolute positioning of branch selector */
        }}
        .message.system {{ background: #f0f0f0; }}
        .message.user {{ background: #f0f7ff; }}
        .message.assistant {{ background: white; }}
        .message-header {{
            margin-bottom: 10px;
            font-size: 0.9em;
            color: #666;
        }}
        .branch-selector {{
            position: absolute;
            bottom: 15px;
            right: 15px;
        }}
        select {{
            padding: 4px;
            border-radius: 4px;
            border: 1px solid #ccc;
        }}
        .thumbnail {{
            max-width: 200px;
            max-height: 200px;
            cursor: pointer;
            margin: 10px 0;
        }}
        .current {{ border-left: 4px solid #007bff; }}
        pre {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }}
        code {{
            font-family: monospace;
            background: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        .footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: #fff;
            padding: 10px;
            text-align: center;
            border-top: 1px solid #eee;
        }}

        .footer {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 20px;
        }}

        .global-branch-selector select {{
            padding: 5px;
            border-radius: 4px;
            border: 1px solid #ccc;
        }}

        .theme-toggle button {{
            background: none;
            border: none;
            cursor: pointer;
            font-size: 1.2em;
            padding: 5px;
        }}

        /* Dark mode styles */
        body.dark-mode {{
            background: #222;
            color: #fff;
        }}

        body.dark-mode .message {{
            background: #333;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }}

        body.dark-mode .message.system {{ background: #2a2a2a; }}
        body.dark-mode .message.user {{ background: #1a2635; }}
        body.dark-mode .message.assistant {{ background: #333; }}

        body.dark-mode .footer {{
            background: #333;
            border-top: 1px solid #444;
        }}

        body.dark-mode select {{
            background: #444;
            color: #fff;
            border-color: #555;
        }}

        body.dark-mode code {{
            background: #444;
        }}

        body.dark-mode pre {{
            background: #444;
        }}

        {custom_css}
    </style>
</head>
<body>
    <h1>{display_title}</h1>
    <div class="info">
        {author_info}
        {f"{tools_info}<br>" if tools_info else ""}
    </div>
    <div id="chat-container"></div>
    <div class="footer">
        <div class="global-branch-selector">
            <select id="globalBranchSelect">
                <!-- Will be populated via JavaScript -->
            </select>
        </div>
        <span class="footer-text">
            made with <a href="https://github.com/abhimanyupallavisudhir/chit">chit</a> by <a href="https://abhimanyu.io">Abhimanyu Pallavi Sudhir</a>
        </span>
        <div class="theme-toggle">
            <button id="themeToggle">🌙</button>
        </div>
    </div>
    <script>
        // Very first thing - basic logging
        console.log('Script started');

        const chatData = {data_str};

        console.log('Data parsed successfully');
        
        marked.setOptions({{ breaks: true, gfm: true }});

        function renderContent(content) {{
            if (typeof content === 'string') return marked.parse(content);
            
            let html = '';
            for (const item of content) {{
                if (item.type === 'text') {{
                    html += marked.parse(item.text);
                }} else if (item.type === 'image_url') {{
                    const url = item.image_url.url;
                    html += `<img src="${{url}}" class="thumbnail" onclick="window.open(this.src, '_blank')" alt="Click to view full size">`;
                }}
            }}
            return html;
        }}

        function getMessagesFromRoot(startId) {{
            console.log('getMessagesFromRoot called with startId:', startId);
            let messages = [];
            let currentId = startId;
            
            // First go back to root
            while (currentId) {{
                const msg = chatData.messages[currentId];
                console.log('Processing message:', msg);
                messages.unshift(msg);
                currentId = msg.parent_id;
            }}
            
            console.log('Messages from root:', messages);
            return messages;
        }}

        function getCompleteMessageChain(startId) {{
            console.log('getCompleteMessageChain called with startId:', startId);
            let messages = getMessagesFromRoot(startId);
            
            // Now follow home_branches forward
            let currentMsg = messages[messages.length - 1];
            console.log('Starting forward traversal from:', currentMsg);
            
            while (currentMsg) {{
                // Get the next message following home_branch
                const children = currentMsg.children;
                const homeBranch = currentMsg.home_branch;
                const nextId = children[homeBranch];
                
                console.log('Current message:', currentMsg.id);
                console.log('Home branch:', homeBranch);
                console.log('Children:', children);
                console.log('Next ID:', nextId);
                
                if (!nextId) break;  // Stop if no child on home_branch
                
                currentMsg = chatData.messages[nextId];
                messages.push(currentMsg);
            }}
            
            console.log('Final message chain:', messages);
            return messages;
        }}

        function onBranchSelect(messageId, selectedBranch) {{
            const msg = chatData.messages[messageId];
            const childId = msg.children[selectedBranch];
            
            if (!childId) return;
            
            chatData.current_id = childId;
            renderMessages();
        }}

        function renderMessages() {{
            console.log('renderMessages called');
            console.log('chatData:', chatData);
            console.log('current_id:', chatData.current_id);
            
            const container = document.getElementById('chat-container');
            container.innerHTML = '';
            
            const messages = getCompleteMessageChain(chatData.current_id);
            console.log('Messages to render:', messages);
            
            messages.forEach(msg => {{
                console.log('Rendering message:', msg);
                const div = document.createElement('div');
                div.className = `message ${{msg.message.role}} ${{msg.id === chatData.current_id ? 'current' : ''}}`;
                
                let branchHtml = '';
                if (msg.children && Object.keys(msg.children).length > 0) {{
                    const branches = Object.entries(msg.children)
                        .filter(([_, childId]) => childId !== null);
                    
                    if (branches.length > 0) {{
                        const options = branches
                            .map(([branch, childId]) => 
                                `<option value="${{branch}}" ${{childId === messages[messages.indexOf(msg) + 1]?.id ? 'selected' : ''}}>${{branch}}</option>`)
                            .join('');
                        
                        branchHtml = `
                            <div class="branch-selector">
                                <select onchange="onBranchSelect('${{msg.id}}', this.value)" 
                                        ${{branches.length === 1 ? 'disabled' : ''}}>
                                    ${{options}}
                                </select>
                            </div>
                        `;
                    }}
                }}
                
                div.innerHTML = `
                    <div class="message-header">
                        <span>${{msg.message.role}} (${{msg.id}})</span>
                    </div>
                    <div class="message-content">
                        ${{renderContent(msg.message.content)}}
                    </div>
                    ${{branchHtml}}
                `;
                
                container.appendChild(div);
            }});
            
            MathJax.typeset();
        }}

        function getAllBranches() {{
            const branches = new Set();
            Object.values(chatData.messages).forEach(msg => {{
                Object.keys(msg.children).forEach(branch => {{
                    branches.add(branch);
                }});
            }});
            return Array.from(branches);
        }}

        function getBranchHierarchy() {{
            // Create a mapping of branch to its immediate parent branch
            const hierarchy = new Map();
            const visited = new Set();

            function traverse(messageId) {{
                if (visited.has(messageId)) return;
                visited.add(messageId);

                const msg = chatData.messages[messageId];
                const homeBranch = msg.home_branch;

                // Check each child branch
                Object.entries(msg.children).forEach(([childBranch, childId]) => {{
                    if (childId && childBranch !== homeBranch) {{
                        // This branch diverges here
                        hierarchy.set(childBranch, homeBranch);
                    }}
                    if (childId) {{
                        traverse(childId);
                    }}
                }});
            }}

            traverse(chatData.root_id);
            return hierarchy;
        }}

        function renderGlobalBranchSelector() {{
            const select = document.getElementById('globalBranchSelect');
            const branchHierarchy = getBranchHierarchy();
            
            // Get all branches
            const allBranches = new Set(['master']);
            Object.values(chatData.messages).forEach(msg => {{
                Object.keys(msg.children).forEach(branch => {{
                    allBranches.add(branch);
                }});
            }});
            
            // Track which branches have been processed
            const processedBranches = new Set();
            
            // Find top-level branches (those that aren't children of any other branch)
            const topLevelBranches = Array.from(allBranches).filter(branch => {{
                // A branch is top-level if it's not a child in the hierarchy
                return branch === 'master' || !Array.from(branchHierarchy.values()).includes(branch);
            }});
            
            function getChildren(parentBranch) {{
                return Array.from(branchHierarchy.entries())
                    .filter(([_, parent]) => parent === parentBranch)
                    .map(([child, _]) => child);
            }}
            
            function renderBranch(branch, level) {{
                if (processedBranches.has(branch)) return ''; // Skip if already processed
                processedBranches.add(branch);
                
                const indent = '&nbsp;'.repeat(level * 2);
                const selected = branch === chatData.current_branch ? 'selected' : '';
                const option = `<option value="${{branch}}" ${{selected}}>${{indent}}${{branch}}</option>`;
                
                let html = option;
                const children = getChildren(branch);
                children.forEach(child => {{
                    html += renderBranch(child, level + 1);
                }});
                return html;
            }}
            
            select.innerHTML = '';
            topLevelBranches.forEach(branch => {{
                select.innerHTML += renderBranch(branch, 0);
            }});
        }}
        function renderBranchOption(branch, level = 0) {{
            return `<option value="${{branch}}" ${{branch === chatData.current_branch ? 'selected' : ''}}>
                ${{'&nbsp;'.repeat(level * 2)}}${{branch}}
            </option>`;
        }}

        function updateGlobalBranchSelector() {{
            const select = document.getElementById('globalBranchSelect');
            const hierarchy = getBranchHierarchy();
            
            function addBranchOptions(obj, prefix = '', level = 0) {{
                Object.keys(obj).forEach(branch => {{
                    const fullBranch = prefix ? `${{prefix}}_${{branch}}` : branch;
                    select.innerHTML += renderBranchOption(fullBranch, level);
                    addBranchOptions(obj[branch], fullBranch, level + 1);
                }});
            }}
            
            select.innerHTML = '';
            addBranchOptions(hierarchy);
        }}

        function onGlobalBranchSelect(branch) {{
            const branchTip = Object.entries(chatData.messages).find(([_, msg]) => 
                msg.children[branch] === null
            );
            
            if (!branchTip) {{
                console.error(`Could not find tip of branch ${{branch}}`);
                return;
            }}

            chatData.current_id = branchTip[0];
            chatData.current_branch = branch;
            renderMessages();
        }}


        // Theme toggle functionality
        function toggleTheme() {{
            document.body.classList.toggle('dark-mode');
            const button = document.getElementById('themeToggle');
            button.textContent = document.body.classList.contains('dark-mode') ? '☀️' : '🌙';
        }}

        // Add event listeners
        document.getElementById('globalBranchSelect').addEventListener('change', (e) => {{
            onGlobalBranchSelect(e.target.value);
        }});

        document.getElementById('themeToggle').addEventListener('click', toggleTheme);

        // Initialize theme based on configuration
        {"document.body.classList.add('dark-mode'); document.getElementById('themeToggle').textContent = '☀️';" if self.display_config.get("dark", True) else ""}


        // Initial render
        renderMessages();

        // Call this after initial render
        renderGlobalBranchSelector();

    </script>
</body>
</html>
"""

    def log(
        self,
        style: Literal["tree", "forum", "gui"] = "tree",
        mode: Literal["print", "return"] = "print",
    ) -> None | str:
        """
        Generate a visualization of the conversation history.

        style="tree" looks like this:

        ```
        001e1e──ab2839──29239b──f2foif9──f2f2f2 (master)
                      ├─bb2b2b──adaf938 (features)
                      |       └─f2f2f2*──aa837r (design_discussion*)
                      |                        ├ (flask_help)
                      |                        └ (tree_viz_help)
                      └─r228df──f2f2f2 (publishing)
                              └─j38392──b16327 (pypi)
        ```

        style="forum" looks like this:

        ```
        [S] 001e1e: You are a helpful assista...
            [U] ab2839: Hello I am Dr James and I...
                [A] 29239b: Hello Dr James, how can I...
                    [U] f2foif9: I am trying to use the...
                        [A] f2f2f2: Have you tried using... (master)
                [A] bb2b2b: Hello Dr James, I see you...
                    [U] adaf938: Can we implement the featur... (features)
                    [U*] f2f2f2: This is actually a design issue...
                        [A] aa837r: Sure, I'll help you design a React... (design_discussion*)
                            (flask_help)
                            (tree_viz_help)
                [A] r228df: I see you are working on the...
                    [U] f2f2f2: Ok that worked. Now let's pu... (publishing)
                    [U] j38392: How do I authenticate with p...
                        [A] b16327: Since you are working with g... (pypi)
        ```

        style="gui" opens a GUI visualization in the browser.

        Args:
            style (str): Style of visualization ("tree", "forum", "gui")
            mode (str): Whether to print the visualization or return it as a string"
        """
        if mode == "print":
            if style == "tree":
                print(self._log_tree())
            elif style == "forum":
                print(self._log_forum())
            elif style == "gui":
                self.gui()
        elif mode == "return":
            if style == "tree":
                return self._log_tree()
            elif style == "forum":
                return self._log_forum()
            elif style == "gui":
                return self.gui(mode="return")
        else:
            raise ValueError(f"Invalid mode {mode}")

    @classmethod
    def migrate(cls, json_file: str, format: Literal["claude"] = "claude") -> "Chat":
        """ "
        Migrate a conversation from a different format to chit."

        Args:
            json_file (str): Path to the JSON file containing the conversation data.
            format (str): Format of the conversation data. Currently only "claude" is supported.
        """
        if format == "claude":
            from chit.import_claude import import_claude

            return import_claude(json_file)
        else:
            raise NotImplementedError(
                f"Migration from {format} format is not supported"
            )

import tempfile
import webbrowser
from dataclasses import dataclass
from typing import Dict, Optional, Pattern, Any, Literal
from pathlib import Path
import uuid
import json
import re
from litellm import completion, stream_chunk_builder
from chit.images import prepare_image_message

@dataclass
class Message:
    id: str
    message: Dict[str, str]
    children: Dict[str, Optional[str]]  # branch_name -> child_id
    parent_id: Optional[str]
    home_branch: str

    @property
    def heir_id(self):
        return self.children[self.home_branch]

class Chat:
    def __init__(self, model: str = "openrouter/deepseek/deepseek-chat"):
        self.model = model
        self.remote = None
        initial_id = str(uuid.uuid4())
        self.root_id = initial_id  # Store the root message ID

        # Initialize with system message
        self.messages: Dict[str, Message] = {
            initial_id: Message(
                id=initial_id,
                message={"role": "system", "content": "You are a helpful assistant."},
                children={"master": None},
                parent_id=None,
                home_branch="master"
            )
        }
        
        self.current_id = initial_id
        self.current_branch = "master"
        # Track latest message for each branch
        # maps each branch name to the latest message that includes
        # that branch in its children attribute's keys
        self.branch_tips: Dict[str, str] = {"master": initial_id}

    def commit(self, role: str = "user", message: str | None = None, image_path: str | Path | None = None) -> str:
        # check that checked-out message does not already have a child in the checked-out branch
        existing_child_id = self.messages[self.current_id].children[self.current_branch]
        if existing_child_id is not None:
            raise ValueError(f"Current message {self.current_id} already has a child message {existing_child_id} on branch {self.current_branch}")
        
        if role == "user" and message is None and image_path is None:
            raise ValueError("User messages must provide content")
        
        if role == self.messages[self.current_id].message["role"]: # NOTE might remove this check
            raise ValueError("Cannot commit two messages with the same role in a row")
        
        new_id = str(uuid.uuid4())

        if image_path is not None:
            assert role == "user", "Only user messages can include images"
            message = prepare_image_message(message, image_path)

        if role == "assistant" and message is None:
            # Generate AI response
            history = self._get_message_history()
            _response = completion(model=self.model, messages=history, stream=True)
            chunks = []
            for chunk in _response:
                print(chunk.choices[0].delta.content or "", end="")
                chunks.append(chunk)
            response = stream_chunk_builder(chunks, messages=history)
            message = response.choices[0].message.content

        # Create new message
        new_message = Message(
            id=new_id,
            message={"role": role, "content": message},
            children={self.current_branch: None},
            parent_id=self.current_id,
            home_branch=self.current_branch
        )
        
        # Update parent's children
        self.messages[self.current_id].children[self.current_branch] = new_id
        
        # Add to messages dict
        self.messages[new_id] = new_message

        # Update branch tip
        self.branch_tips[self.current_branch] = new_id

        # Update checkout
        self.current_id = new_id

        # return new_message.message["content"]

    def branch(self, branch_name: str, checkout: bool = False) -> None:
        if branch_name in self.branch_tips:
            raise ValueError(f"Branch '{branch_name}' already exists (latest at message {self.branch_tips[branch_name]})")
                
        self.messages[self.current_id].children[branch_name] = None
        self.branch_tips[branch_name] = self.current_id
        if checkout:
            old_id = self.current_id
            self.checkout(branch_name=branch_name)
            assert self.current_id == old_id # since we just created the branch, it should be the same as before

    def _resolve_forward_path(self, branch_path: list[str], start_id: Optional[str] = None) -> str:
        """Follow a path of branches forward from start_id (or current_id if None)"""
        current = start_id if start_id is not None else self.current_id
        
        for branch in branch_path:
            current_msg = self.messages[current]
            if branch not in current_msg.children:
                raise KeyError(f"Branch '{branch}' not found in message {current}")
            
            next_id = current_msg.children[branch]
            if next_id is None:
                raise IndexError(f"Branch '{branch}' exists but has no message in {current}")
                
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

    def _resolve_positive_index(self, index: int) -> str:
        """Convert positive index to message ID by following master branch from root"""
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

    def checkout(self, message_id: Optional[str | int | list[str]] = None, branch_name: Optional[str] = None) -> None:
        if message_id is not None:
            if isinstance(message_id, int):
                if message_id >= 0:
                    message_id = self._resolve_positive_index(message_id)
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
            self.current_id = self.branch_tips[branch_name]
            self.current_branch = branch_name
        else:
            self.current_branch = self.messages[self.current_id].home_branch

    def _get_message_history(self) -> list[dict[str, str]]:
        """Reconstruct message history from current point back to root"""
        history = []
        current = self.current_id
        
        while current is not None:
            msg = self.messages[current]
            history.insert(0, msg.message)
            current = msg.parent_id
            
        return history

    def push(self) -> None:
        """Save chat history to configured remote"""
        if self.remote is None:
            raise ValueError("No remote configured. Set chat.remote first.")
            
        data = {
            "model": self.model,
            "messages": {k: vars(v) for k, v in self.messages.items()},
            "current_id": self.current_id,
            "current_branch": self.current_branch,
            "root_id": self.root_id,
            "branch_tips": self.branch_tips
        }
        with open(self.remote, 'w') as f:
            json.dump(data, f)


    def __getitem__(self, key: str | int | list[str] | slice) -> Message | list[Message]:
        # Handle string indices (commit IDs)
        if isinstance(key, str):
            if key not in self.messages:
                raise KeyError(f"Message {key} does not exist")
            return self.messages[key]
            
        # Handle integer indices
        if isinstance(key, int):
            if key >= 0:
                return self.messages[self._resolve_positive_index(key)]
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
                    start_id = self._resolve_positive_index(key.start)
                else:
                    start_id = self._resolve_negative_index(key.start)
            elif isinstance(key.start, list):
                start_id = self._resolve_forward_path(key.start)
            else:
                start_id = key.start
                
            stop_id = None
            if isinstance(key.stop, int):
                if key.stop >= 0:
                    stop_id = self._resolve_positive_index(key.stop)
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
    def clone(cls, remote: str) -> 'Chat':
        """Create new Chat instance from remote file"""
        with open(remote, 'r') as f:
            data = json.load(f)
        
        chat = cls(model=data["model"])
        chat.remote = remote  # Set remote automatically when cloning
        chat.messages = {k: Message(**v) for k, v in data["messages"].items()}
        chat.current_id = data["current_id"]
        chat.current_branch = data["current_branch"]
        chat.root_id = data["root_id"]
        chat.branch_tips = data["branch_tips"]
        return chat

    def rm(self, branch_name: str) -> None:
        """Remove all messages associated with a branch."""
        # Check if we're trying to remove current branch or home branch
        if (branch_name == self.current_branch or 
            branch_name == self.messages[self.current_id].home_branch):
            raise ValueError("KalidasaError: attempted to remove the currently checked out branch or the home branch of the currently checked out message")

        # First pass: identify messages to delete and clean up their parent references
        to_delete = set()
        parent_cleanups = []  # List of (parent_id, msg_id) tuples to clean up
        
        for msg_id, msg in self.messages.items():
            if msg.home_branch == branch_name:
                to_delete.add(msg_id)
                if msg.parent_id is not None:
                    parent_cleanups.append((msg.parent_id, msg_id))

        # Clean up parent references
        for parent_id, msg_id in parent_cleanups:
            if parent_id in self.messages:  # Check parent still exists
                parent = self.messages[parent_id]
                # Find and remove this message from any branch in parent's children
                for branch, child_id in list(parent.children.items()):  # Create list copy to modify during iteration
                    if child_id == msg_id:
                        parent.children[branch] = None

        # Finally delete the messages
        for msg_id in to_delete:
            if msg_id in self.messages:  # Check message still exists
                del self.messages[msg_id]

        # Remove from branch_tips if present
        if branch_name in self.branch_tips:
            del self.branch_tips[branch_name]

    def mv(self, branch_name_old: str, branch_name_new: str) -> None:
        """Rename a branch throughout the tree."""
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

    def find(self, 
             pattern: str | Pattern,
             *,
             case_sensitive: bool = False,
             roles: Optional[list[str]] = None,
             max_results: Optional[int] = None,
             regex: bool = False,
             context: int = 0  # Number of messages before/after to include
             ) -> list[dict[str, Message | list[Message]]]:
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
            if (roles is None or message.message["role"] in roles) and \
               pattern.search(message.message["content"]):
                
                # Get context if requested
                context_messages = []
                if context > 0:
                    start_idx = max(0, len(message_sequence) - context - 1)
                    end_idx = min(len(message_sequence) + context, len(message_sequence))
                    context_messages = message_sequence[start_idx:end_idx]
                    context_messages.remove(message)  # Don't include the match itself in context
                
                results.append({
                    'match': message,
                    'context': context_messages
                })
                
                if max_results and len(results) >= max_results:
                    break
            
            # Move to next message on master branch
            current_id = message.children.get("master")
            
        return results

    def _process_commit_id(self, commit_id: str):
        """Helper function for Chat.log()"""
        commit = self.messages[commit_id]
        commit_id_proc = commit_id[:6]
        role = commit.message['role']
        prefix = f"[{role[0].upper()}{'*' if commit_id == self.current_id else '_'}]"
        commit_id_proc = prefix + commit_id_proc
        return commit_id_proc
    
    def _process_branch_name(self, branch_name: str):
        """Helper function for Chat.log()"""
        if branch_name == self.current_branch:
            return f' ({branch_name}*)'
        return f' ({branch_name})'

    def _log_tree_draw_from(self, frontier_id: str, branch_name: str) -> list[str]:
        """Helper function for Chat.log()"""
        log_lines: list[str] = []
        log_lines.append(self._process_commit_id(frontier_id))
        frontier: Message = self.messages[frontier_id]

        horizontal_pos: int = len(log_lines[0]) # position where stuff should be added

        if hasattr(frontier, "heir_id"):
            log_lines[0] += '──'
            if frontier.heir_id is None:
                log_lines[0] += self._process_branch_name(branch_name)
            else:
                subtree: list[str] = self._log_tree_draw_from(frontier.heir_id, frontier.home_branch)
                # we would like to just append subtree to the current log
                # but it's actually multiple lines that need to get appended 
                # to the right propositions
                indent: int = len(log_lines[0])
                log_lines[0] += subtree[0]
                for subtree_line in subtree[1:]:
                    log_lines.append(' ' * indent + subtree_line)

        for child_branch, child_id in frontier.children.items():
            if child_branch == frontier.home_branch:
                # already processed the heir
                continue
            else:
                for i in range(len(log_lines)):
                    if i == 0:
                        continue
                    line = log_lines[i]
                    if line[horizontal_pos] == '└': # no longer the final branch
                        line = line[:horizontal_pos] + '├' + line[horizontal_pos+1:]
                    if line[horizontal_pos] == ' ': # extend
                        line = line[:horizontal_pos] + '│' + line[horizontal_pos+1:]
                    log_lines[i] = line
                log_lines.append(' ' * horizontal_pos + '└─')
                if child_id is None:
                    log_lines[-1] += self._process_branch_name(child_branch)
                else:
                    subtree: list[str] = self._log_tree_draw_from(child_id, child_branch)
                    indent: int = horizontal_pos + 1 # the length of log_lines[-1]
                    log_lines[-1] += subtree[0]
                    for subtree_line in subtree[1:]:
                        log_lines.append(' ' * indent + subtree_line)
        
        # if not frontier.children or all(v is None for v in frontier.children.values()):
        #     log_lines[0] += self._process_branch_name(branch_name)
        return log_lines

    def _log_tree(self) -> str:
        """
        Generate a tree visualization of the conversation history, like this:

        001e1e──ab2839──29239b──f2foif9──f2f2f2 (master)
                      ├─bb2b2b──adaf938 (features)
                      |       └─f2f2f2*──aa837r (design_discussion*)
                      |                        ├ (flask_help)
                      |                        └ (tree_viz_help)
                      └─r228df──f2f2f2 (publishing)
                              └─j38392──b16327 (pypi)

        """        
        log_lines: list[str] = self._log_tree_draw_from(self.root_id, 'master')
        res = '\n'.join(log_lines)
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
        content_proc = content_proc[:57] + '...'
        return content_proc
            

    def _log_forum_draw_from(self, frontier_id: str) -> list[str]:
        log_lines: list[str] = []
        frontier: Message = self[frontier_id]
        log_lines.append(f"{self._process_commit_id(frontier_id)}: {self._process_message_content(frontier.message['content'])}")
        # show heir first
        if hasattr(frontier, "heir_id"):
            if frontier.heir_id is None:
                log_lines[0] += self._process_branch_name(frontier.home_branch)
            else:
                subtree: list[str] = self._log_forum_draw_from(frontier.heir_id) # recurse
                subtree_ = [' ' * 4 + line for line in subtree] # indent
                log_lines.extend(subtree_)
        for child_branch, child_id in frontier.children.items():
            if child_branch == frontier.home_branch:
                continue # already processed the heir
            elif child_id is None:
                log_lines.append(' ' * 4 + self._process_branch_name(child_branch))
            else:
                subtree: list[str] = self._log_forum_draw_from(child_id) # recurse
                subtree_ = [' ' * 4 + line for line in subtree] # indent
                log_lines.extend(subtree_)
        # if not frontier.children or all(v is None for v in frontier.children.values()):
        #     log_lines[0] += self._process_branch_name(frontier.home_branch)
        return log_lines
        

    def _log_forum(self) -> str:
        """
        Generate a forum-style visualization of the conversation history, like this:

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
        """
        log_lines: list[str] = self._log_forum_draw_from(self.root_id)
        res = "\n".join(log_lines)
        return res


    def _gui(self, file_path: Optional[str | Path] = None) -> None:
        """
        Create and open an interactive visualization of the chat tree.
        
        Args:
            file_path: Optional path where the HTML file should be saved.
                    If None, creates a temporary file instead.
        """
        html_content = self._generate_viz_html()
        
        if file_path is not None:
            # Convert to Path object if string
            path = Path(file_path)
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write the file
            path.write_text(html_content)
            # Open in browser
            webbrowser.open(f'file://{path.absolute()}')
        else:
            # Original temporary file behavior
            with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False) as f:
                f.write(html_content)
                temp_path = f.name
            webbrowser.open(f'file://{temp_path}')
    
    def _prepare_messages_for_viz(self) -> Dict[str, Any]:
        """Convert messages to a format suitable for visualization."""
        return {
            'messages': {k: {
                'id': m.id,
                'message': m.message,
                'children': m.children,
                'parent_id': m.parent_id,
                'home_branch': m.home_branch
            } for k, m in self.messages.items()},
            'current_id': self.current_id,
            'root_id': self.root_id
        }
    
    def _generate_viz_html(self) -> str:
        """Generate the HTML for visualization."""
        data = self._prepare_messages_for_viz()
        
        return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chat Visualization</title>
        <meta charset="UTF-8">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js"></script>
        <style>
            body {{
                font-family: system-ui, -apple-system, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }}
            .message {{
                margin: 20px 0;
                padding: 15px;
                border-radius: 10px;
                background: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .message.system {{ background: #f0f0f0; }}
            .message.user {{ background: #f0f7ff; }}
            .message.assistant {{ background: white; }}
            .message-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
                font-size: 0.9em;
                color: #666;
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
        </style>
    </head>
    <body>
        <div id="chat-container"></div>

        <script>
            const chatData = {json.dumps(data)};
            
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
                let messages = [];
                let currentId = startId;
                
                // First go back to root
                while (currentId) {{
                    const msg = chatData.messages[currentId];
                    messages.unshift(msg);
                    currentId = msg.parent_id;
                }}
                
                return messages;
            }}

            function getCompleteMessageChain(startId) {{
                let messages = getMessagesFromRoot(startId);
                
                // Now follow home_branches forward
                let currentMsg = messages[messages.length - 1];
                while (currentMsg) {{
                    // Get the next message following home_branch
                    const children = currentMsg.children;
                    const homeBranch = currentMsg.home_branch;
                    const nextId = children[homeBranch];
                    
                    if (!nextId) break;  // Stop if no child on home_branch
                    
                    currentMsg = chatData.messages[nextId];
                    messages.push(currentMsg);
                }}
                
                return messages;
            }}

            function onBranchSelect(messageId, selectedBranch) {{
                console.log('Branch selected:', messageId, selectedBranch);
                
                const msg = chatData.messages[messageId];
                const childId = msg.children[selectedBranch];
                
                if (!childId) return;
                
                // Get complete chain including all home_branch children
                chatData.current_id = childId;
                renderMessages();
            }}

            function renderMessages() {{
                console.log('Rendering messages, current_id:', chatData.current_id);
                
                const container = document.getElementById('chat-container');
                container.innerHTML = '';
                
                const messages = getCompleteMessageChain(chatData.current_id);
                console.log('Messages to render:', messages.map(m => m.id));
                
                messages.forEach(msg => {{
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
                                <select onchange="onBranchSelect('${{msg.id}}', this.value)" 
                                        ${{branches.length === 1 ? 'disabled' : ''}}>
                                    ${{options}}
                                </select>
                            `;
                        }}
                    }}                    
                    div.innerHTML = `
                        <div class="message-header">
                            <span>${{msg.message.role}}</span>
                            ${{branchHtml}}
                        </div>
                        <div class="message-content">
                            ${{renderContent(msg.message.content)}}
                        </div>
                    `;
                    
                    container.appendChild(div);
                }});
                
                MathJax.typeset();
            }}

            // Initial render
            renderMessages();
        </script>
    </body>
    </html>
    """

    def log(self, style:Literal["tree", "forum", "gui"]="tree"):
        if style == "tree":
            print(self._log_tree())
        elif style == "forum":
            print(self._log_forum())
        elif style == "gui":
            self._gui()
from dataclasses import dataclass
from typing import Dict, Optional
import uuid
import json
from litellm import completion

@dataclass
class Message:
    id: str
    message: Dict[str, str]
    children: Dict[str, Optional[str]]  # branch_name -> child_id
    parent_id: Optional[str]
    home_branch: str

class Chat:
    def __init__(self, model: str = "openrouter/deepseek/deepseek-chat"):
        self.model = model
        self.remote = None
        initial_id = str(uuid.uuid4())
        
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

    def commit(self, role: str = "user", message: str | None = None) -> str:
        if role == "user" and message is None:
            raise ValueError("User messages must provide content")
        
        new_id = str(uuid.uuid4())
        
        if role == "assistant" and message is None:
            # Generate AI response
            history = self._get_message_history()
            response = completion(model=self.model, messages=history)
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
        
        # Update checkout
        self.current_id = new_id

        return new_message.message["content"]

    def branch(self, branch_name: str) -> None:
        if branch_name in self.messages[self.current_id].children:
            raise ValueError(f"Branch {branch_name} already exists at this point")
        
        self.messages[self.current_id].children[branch_name] = None

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

    def checkout(self, message_id: Optional[str | int] = None, branch_name: Optional[str] = None) -> None:
        if message_id is not None:
            if isinstance(message_id, int):
                if message_id >= 0:
                    raise ValueError("Positive integer indices not supported")
                message_id = self._resolve_negative_index(message_id)
            elif message_id not in self.messages:
                raise ValueError(f"Message {message_id} does not exist")
            self.current_id = message_id
            
        if branch_name is not None:
            if branch_name not in self.messages[self.current_id].children:
                raise ValueError(f"Branch {branch_name} does not exist at message {self.current_id}")
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
            "current_branch": self.current_branch
        }
        with open(self.remote, 'w') as f:
            json.dump(data, f, indent=4)

    @property
    def current_message(self) -> Message:
        return self.messages[self.current_id]

    def __getitem__(self, key: str | int | slice) -> Message | list[Message]:
        # Handle string indices (commit IDs)
        if isinstance(key, str):
            if key not in self.messages:
                raise KeyError(f"Message {key} does not exist")
            return self.messages[key]
            
        # Handle negative integer indices
        if isinstance(key, int):
            if key >= 0:
                raise ValueError("Positive integer indices not supported")
            return self.messages[self._resolve_negative_index(key)]
            
        # Handle slices
        if isinstance(key, slice):
            if key.step is not None:
                raise ValueError("Step is not supported in slicing")
                
            # Convert start/stop to message IDs if they're negative integers
            start_id = (self._resolve_negative_index(key.start) 
                       if isinstance(key.start, int) else key.start)
            stop_id = (self._resolve_negative_index(key.stop) 
                      if isinstance(key.stop, int) else key.stop)
            
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
                
            return result
            
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
        return chat
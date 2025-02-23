from dataclasses import dataclass
from typing import Dict, Optional
import uuid
import json
from litellm import completion

ALIASES = {
    "deepseek": "openrouter/deepseek/deepseek-chat",
    "claude": "openrouter/anthropic/claude-3.5-sonnet",
    "oldclaude": "openrouter/anthropic/claude-3.5-sonnet-20240620",
}

@dataclass
class Message:
    id: str
    message: Dict[str, str]
    children: Dict[str, Optional[str]]  # branch_name -> child_id
    parent_id: Optional[str]
    home_branch: str

class Chat:
    def __init__(self, model: str = "openrouter/deepseek/deepseek-chat"):
        self.model = ALIASES.get(model, model)
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
            message = response
        
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
        
        # return new_id

    def branch(self, branch_name: str) -> None:
        if branch_name in self.messages[self.current_id].children:
            raise ValueError(f"Branch {branch_name} already exists at this point")
        
        self.messages[self.current_id].children[branch_name] = None

    def checkout(self, message_id: Optional[str] = None, branch_name: Optional[str] = None) -> None:
        if message_id is not None:
            if message_id not in self.messages:
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

    def push(self, filename: str) -> None:
        """Save chat history to JSON file"""
        data = {
            "model": self.model,
            "messages": {k: vars(v) for k, v in self.messages.items()},
            "current_id": self.current_id,
            "current_branch": self.current_branch
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    @classmethod
    def clone(cls, filename: str) -> 'Chat':
        """Create new Chat instance from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        chat = cls(model=data["model"])
        chat.messages = {k: Message(**v) for k, v in data["messages"].items()}
        chat.current_id = data["current_id"]
        chat.current_branch = data["current_branch"]
        return chat

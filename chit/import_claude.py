from pathlib import Path
import json
from chit import Chat, ChitMessage
from chit.images import prepare_image_message

from litellm.types.utils import (
    # ModelResponse,
    ChatCompletionMessageToolCall,
    Function,
)


def import_claude(claude_export_path: str | Path, system_prompt: str = None) -> Chat:
    """
    Import a Claude chat export into a chit.Chat object
    
    Args:
        claude_export_path: Path to Claude JSON export file
        system_prompt: Optional system prompt to use (if not provided, will use "You are a helpful assistant.")
    
    Returns:
        chit.Chat instance with imported conversation
    """
    # Load the Claude export file
    with open(claude_export_path, 'r', encoding='utf-8') as f:
        claude_data = json.load(f)
    
    # Create a new Chat instance with default or provided system prompt
    chat = Chat(model="anthropic/claude-3-opus-20240229")  # Use Claude model by default
    
    # If user provided a custom system prompt, update the root message
    if system_prompt:
        chat.messages[chat.root_id].message["content"] = system_prompt
        
    # Create a mapping from Claude UUIDs to our short IDs
    uuid_to_id_map = {"00000000-0000-4000-8000-000000000000": chat.root_id}
    
    # Sort messages by parent relationships to ensure we process them in order
    messages = claude_data["chat_messages"]
    
    # Process messages
    processed_branches = set(["master"])  # Track branches we've already created
    branch_counter = 1  # Counter for generating unique branch names
    
    # First pass - create ID mapping and identify message hierarchy
    for msg in messages:
        # Generate a new short ID for this message
        new_id = chat._generate_short_id()
        uuid_to_id_map[msg["uuid"]] = new_id
    
    # Second pass - process messages and build structure
    for msg in messages:
        claude_uuid = msg["uuid"]
        parent_uuid = msg["parent_message_uuid"]
        
        # Skip if this is the system message (UUID 00000000-0000-4000-8000-000000000000)
        if parent_uuid == "00000000-0000-4000-8000-000000000000":
            parent_id = chat.root_id
        else:
            # If parent isn't in our map yet, skip for now
            if parent_uuid not in uuid_to_id_map:
                continue
            parent_id = uuid_to_id_map[parent_uuid]
        
        message_id = uuid_to_id_map[claude_uuid]
        
        # Determine if we need to create a new branch
        parent_msg = chat.messages[parent_id]
        
        # Check if we need to create a new branch
        need_new_branch = True
        branch_name = "master"  # Default branch
        
        # Check if we can use an existing branch from the parent
        for branch, child_id in parent_msg.children.items():
            if child_id is None:
                # Found an empty branch we can use
                need_new_branch = False
                branch_name = branch
                break
                
        # Create a new branch if needed
        if need_new_branch and branch_name == "master":
            # We need to create a new branch
            branch_name = f"branch_{branch_counter}"
            branch_counter += 1
            # Add the branch to the parent's children
            parent_msg.children[branch_name] = None
            processed_branches.add(branch_name)
        
        # Extract message content
        if msg["sender"] == "human":
            role = "user"
        else:  # assistant
            role = "assistant"
            
        # Process message content
        content = _process_claude_content(msg["content"])
        
        # Extract tool calls if present
        tool_calls = None
        if role == "assistant":
            tool_calls = _extract_claude_tools(msg["content"])
        
        # Create new message
        new_message = ChitMessage(
            id=message_id,
            message={"role": role, "content": content},
            children={branch_name: None},  # Will connect any children in subsequent iterations
            parent_id=parent_id,
            home_branch=branch_name,
            tool_calls=tool_calls
        )
        
        # Update parent's children
        parent_msg.children[branch_name] = message_id
        
        # Update branch tip
        chat.branch_tips[branch_name] = message_id
        
        # Add to messages dict
        chat.messages[message_id] = new_message
    
    # Set the current checkout to the leaf message specified in the Claude export
    if "current_leaf_message_uuid" in claude_data and claude_data["current_leaf_message_uuid"] in uuid_to_id_map:
        leaf_id = uuid_to_id_map[claude_data["current_leaf_message_uuid"]]
        # Find the branch this message is on
        leaf_message = chat.messages[leaf_id]
        leaf_branch = leaf_message.home_branch
        # Checkout this branch and message
        chat.current_id = leaf_id
        chat.current_branch = leaf_branch
    
    return chat

def _process_claude_content(content_list) -> str:
    """
    Process Claude's content format into a plain text string for chit.
    
    Args:
        content_list: List of content items from Claude's export
        
    Returns:
        Processed text content
    """
    if not content_list:
        return ""
    
    full_text = ""
    
    for item in content_list:
        if item["type"] == "text":
            full_text += item["text"]
        # Skip non-text content for now - we'll extract it separately for tools
    
    return full_text

def _extract_claude_tools(content_list) -> list[ChatCompletionMessageToolCall] | None:
    """
    Extract tool calls from Claude's content format into the format chit expects.
    
    Args:
        content_list: List of content items from Claude's export
        
    Returns:
        List of tool calls in litellm format, or None if no tools were used
    """
    if not content_list:
        return None
    
    tool_calls = []
    current_tool_call = None
    
    for item in content_list:
        if item["type"] == "tool_use":
            # Start a new tool call
            tool_id = f"call_{len(tool_calls)}"
            
            if item["name"] == "artifacts":
                # Handle Claude artifacts as a special case
                tool_input = item["input"]
                function_name = "artifact_" + (tool_input.get("command", "create"))
                
                current_tool_call = ChatCompletionMessageToolCall(
                    id=tool_id,
                    type="function",
                    function=Function(
                        name=function_name,
                        arguments=json.dumps(tool_input)
                    )
                )
                tool_calls.append(current_tool_call)
            
        elif item["type"] == "tool_result":
            # Tool results will be processed separately in our system
            pass
    
    return tool_calls if tool_calls else None
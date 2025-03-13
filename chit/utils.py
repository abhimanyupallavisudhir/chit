from pathlib import Path
import chit.config

def wordcel(*args, **kwargs):
    """I can't get logging to print things in the right place in a notebook.
    
    Just print, but only if VERBOSE is True.
    """
    if chit.config.VERBOSE:
        print(*args, **kwargs)


def annoy(prompt: str) -> bool:
    """Prompt the user to confirm an action."""
    if not chit.config.FORCE:
        response = input(f"{prompt} (y/n) ")
        return response.lower() == "y"
    return True


def read(file_path: str | Path) -> str:
    """Read a file and return its contents.

    Quite useful for passing text files to the LLM.
    """
    with open(file_path, "r") as f:
        return f.read()
    
def textput():
    """Create a text area for user input in a Jupyter notebook.
    
    Usage:

    ```
    prompt = textput()
    
    # ... write your prompt in the text area that appears, 
    # no need to worry about clicking any submit buttons,
    # it's dynamic ...
    # 
    # ... then in a new cell ...

    chat.commit(prompt.value)
    chat.commit()
    ```

    """

    import ipywidgets as widgets
    from IPython.display import display
        
    # Create text area and button
    text_area = widgets.Textarea(
        placeholder='Type your message here...',
        layout=widgets.Layout(width='100%', height='200px')
    )
    
    # Display widgets
    display(text_area)

    return text_area

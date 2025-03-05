from pathlib import Path
import chit.config

def cprint(*args, **kwargs):
    """I can't get logging to print things in the right place in a notebook."""
    if chit.config.VERBOSE:
        print(*args, **kwargs)


def cconfirm(prompt: str) -> bool:
    """Prompt the user to confirm an action."""
    if not chit.config.FORCE:
        response = input(f"{prompt} (y/n) ")
        return response.lower() == "y"
    return True


def read(file_path: str | Path) -> str:
    with open(file_path, "r") as f:
        return f.read()
    
def textput():
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

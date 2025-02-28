# chit

`chit` is a git-analogous system for managing LLM chat conversations in a Jupyter notebook. Install as `pip install chitgpt`.

The `chit.Chat` class has methods:

- `commit()` for adding new messages (either user or assistant). For creating an assistant message, the message path leading up to the current checked-out message is sent to the LLM.
- `branch()` for creating a new branch at the current checked-out message
- `checkout()` for changing the checkout message. 
- `push()` for dumping data to a json file we call the `remote`
- `clone()` a classmethod for initializing a `chit.Chat` object from a json file
- sensible indexing and slicing
- `rm()` for removing a branch
- `mv()` for moving a branch
- `find()` for finding in conversation history
- `log()` for creating
- `viz()` for creating a (non-interactive) gui output of the conversation similar to a classic LLM interface

We use [litellm](https://github.com/BerriAI/litellm) for the LLM completions, so use their model naming conventions. Vision is supported, including images from the clipboard like so: `chat.commit("user", "Analyze this image.", image_path = 'CB')`.

See [example.ipynb](example.ipynb) for some demonstration. 
# Multimodal (Decoder) RAG
## Todos
- System (chat/ rag) prompt
- Autoregressive Transfomer response
  
# Installation
1. Clone the repsitory:

```
git clone git@github.com:CKeibel/FHSWF-deep-learning.git
```

2. Checkout directory:
```
cd FHSWF-deep-learning
```

### Installing with pip

1. Create a virtual environment:
```
python -m venv .venv
```

2. Activate the newly create virtual env named "venv":
```
source .venv/bin/activate
```
Now `(.venv)` should be displayed in front of your command prompt.

3. Install project dependencies with **pip**:

```
python -m pip install -e .
```

### Poetry

Install **poetry**
```
pip install poetry
```

Install project dependencies with **poetry**:
```
poetry install
```

## Usage
Stat the application via `python main.py`.

When starting up, two urls will be available to access the interface. Use the **local url** when you are working on your local machine. If the app runs on a remote cluster (e.g. the fh-swf cluster) use the **public url**.

**IMPORTANT**:
When you start the app on the *fh-swf cluster* make sure that your “current working directory” is set correctly in *vscode*. This is absolutely necessary to read the `models.yml` when starting the app.
```
# launch.json
{
    "configurations": [
        {
            "name": "App",
            "type": "python",
            "request": "launch",
            ...
            "cwd": "/home/<USER>/FHSWF-deep-learning/", # set <USER>
            "program": "main.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            ...
        }
    ]
}
```

# Dev
## pre-commit
Install `pre-commit`
```
pip install pre-commit
```

Install **pre-commit hooks**
```
pre-commit install
-------------------
>> pre-commit installed at .git/hooks/pre_commit
```

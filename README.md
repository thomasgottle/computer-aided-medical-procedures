# computer-aided-medical-procedures
Computer Aided Medical Procedures 1 - AI in Computational Medicine &amp; Surgery

## Practicals

Each practical lives in its own folder with its own `README.md`, `requirements.txt` and notebooks:

| Folder | Topic |
| --- | --- |
| `registration_practical/` | Point, intensity, deformable and learning-based image registration |

## Setting up a practical's environment

Each practical has its own Python environment inside its folder. Follow the setup section of that practical's README, and **run every setup command from inside the practical's folder** (the one containing `requirements.txt`), not from this repository root:

```sh
cd registration_practical
python3.13 -m venv .venv313
./.venv313/bin/python -m pip install -r requirements.txt
```

If you run the commands from the repository root instead, the environment is created in the wrong place and `pip install -r requirements.txt` fails because no `requirements.txt` exists there. That leaves an empty environment without NumPy.

## Selecting the kernel in VS Code

Install the **Python** and **Jupyter** extensions (Microsoft). When this repository root is open, VS Code may not list environments located inside practical folders, and several environments can share the same name, such as `.venv313 (3.13.x)`.

If a notebook reports `No module named 'numpy'` or the environment is missing from the list:

1. Click the kernel name at the top right → **Select Another Kernel… → Python Environments… → Enter interpreter path…**
2. Enter the full path to the practical's interpreter, e.g. `registration_practical/.venv313/bin/python` (Windows: `registration_practical\.venv313\Scripts\python.exe`).
3. Restart the kernel and check with `import sys; print(sys.executable)`.

Alternatively, open the practical's folder itself (**File → Open Folder…**); VS Code then finds its environment automatically.

If a kernel was running from an environment that has since been deleted, run **Python: Clear Workspace Interpreter Setting** and **Developer: Reload Window** from the Command Palette, then select the kernel again.

# Steps to run
- pip install -r requirements.txt
- huggingface-cli login
  - token: <paylaşılacak>
- First run gdpo.py
- After gdpo.py finished, run dpo.py (or create a copy of the project and run in parallel)
- After both finished, run inference.py to test models
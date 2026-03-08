# AudioEnhancement
This is our audio enhancement library including both single and multichannel algorithms for noise reduction

## Getting Started
First, run the script [scripts/run-setup.bat](scripts/run-setup.bat), It will do the following:
1. Create a virtual environment
2. Install all the required packages
3. Make sure that every time you commit, [requirements.txt](requirements.txt) is updated if there were dependency changes.
4. Will ruff check all your python files and make sure they don't have any errors violate PEP8 – if they do - ruff will fix what it can.

_**NOTE:**_ **ruff might reformat your files in the pre-commit stage, so the commit will fail. Don't worry – 
just stage and commit again, that's normal behavior :)**

**If you want to be absolutely sure that your code is PEP8 compliant, you can run this command:**
```
pre-commit run --all-files
```

## Pulling Changes
whenever you want to pull changes from the main branch, run these commands:
```
git fetch origin
git pull
pip install -r requirements.txt
```
**Enjoy!**

## PR Status Checks
Whenever you request a PR, the following checks will be performed:
1. **Code Formatting and quality**: Ruff will check all Python files for PEP8 compliance – test will fail if any of the files are not PEP8 compliant.
In addition, PyLint will check all Python files for code quality – this test will fail if any of the files have code quality issues.

2. CI that includes:
- Creating a Windows VM with python 3.10
- Creating a virtual environment
- Installing all the required packages
- Checking for requiremnets.txt consistency
- Running all the tests (if you wrote any) and making sure they pass

when requesting a PR you can see the status of these checks with a full log (so you will be able to know why it failed)

## Do not change these files:
Anything that is not a python file – these are used for the automations I mentioned.

**Enjoy!**

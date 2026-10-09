# Windows setup from scratch

For a Windows machine with nothing installed (no Python, no Git). Follow the
steps in order. Most of the 30–45 minutes is downloads.

**Requirements:** Windows 10 or 11, about 2 GB free disk, internet connection.
No Kaggle account needed.

---

## 1. Install Python

Download **Python 3.12** from <https://www.python.org/downloads/windows/>
(choose "Windows installer (64-bit)").

> Use 3.12, not 3.13 or newer. Some packages don't ship Windows wheels for the
> newest versions, so pip falls back to compiling them, which fails without a
> C++ compiler installed.

In the installer:

1. **Tick "Add python.exe to PATH"** at the bottom of the first screen.
   Missing this is what causes the `'python' is not recognized` error later.
2. Click **Install Now**.
3. On the last screen, if **"Disable path length limit"** is shown, click it.

Check it worked. Open **Command Prompt** (press Start, type `cmd`, press Enter):

```
py --version
```

You should see `Python 3.12.x`. If you get an error, see
[Troubleshooting](#troubleshooting).

---

## 2. Install Git

Download from <https://git-scm.com/download/win> and run the installer. Accept
every default.

Skip this step if you are copying the project folder from USB instead of cloning.

---

## 3. Get the project

In Command Prompt, pick a folder and clone:

```
cd %USERPROFILE%\Documents
git clone <your-repository-url> capstone
cd capstone
```

Replace `<your-repository-url>` with your GitHub URL.

**Copying from USB?** Copy the folder to `Documents\capstone`, then
`cd %USERPROFILE%\Documents\capstone`. Delete any `.venv` folder inside it; a
virtual environment built on another machine won't run here.

---

## 4. Create the virtual environment

```
py -m venv .venv
.venv\Scripts\activate
```

Your prompt should now start with `(.venv)`.

> **If you are using PowerShell** instead of Command Prompt and activation is blocked
> with "running scripts is disabled on this system", either use Command Prompt
> instead, or run this once:
> ```
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

---

## 5. Install the packages

```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

About 150 MB of downloads; 3–10 minutes. Installs pandas, scikit-learn,
Streamlit, Plotly, NLTK, and PyArrow.

Then fetch the VADER lexicon:

```
python -c "import nltk; nltk.download('vader_lexicon')"
```

---

## 6. Smoke-test before downloading data

```
python pipeline.py --test
```

Expected output: `self-check ok`.

Then run the full pipeline on the 600-row sample that ships with the repo:

```
python pipeline.py --raw data/Reviews.sample.csv --rows 0 --topics 4
```

If this writes files to `out\` without errors, your setup is good and you can
move on to the real dataset.

---

## 7. Download the real dataset

```
python fetch_data.py
```

Downloads **122 MB** from Stanford SNAP and converts it to `data\Reviews.csv`
(289 MB, 568,454 reviews). 5–15 minutes depending on connection.

No Kaggle account or API token is needed; SNAP is the original source that
Kaggle repackages.

---

## 8. Run the pipeline

Start with a sample to confirm the pipeline runs end to end:

```
python pipeline.py --rows 50000
```

Then the full dataset:

```
python pipeline.py --rows 0 --topics 12
```

The full run is **3–6 minutes** and needs about **4 GB of free RAM**. It prints
the ingestion funnel, accuracy report, and ranked opportunities, then writes
the parquet files into `out\`.

---

## 9. Start the dashboard

```
streamlit run app.py
```

Your browser opens at <http://localhost:8501>. Keep the Command Prompt window
open while you use the dashboard; closing it stops the server.

Press `Ctrl+C` in Command Prompt to stop.

---

## Coming back later

Each new Command Prompt session needs the virtual environment reactivated:

```
cd %USERPROFILE%\Documents\capstone
.venv\Scripts\activate
streamlit run app.py
```

Steps 1–7 only need to be done once.

---

## Troubleshooting

**`'python' is not recognized as an internal or external command`**
Python is not on PATH. Either reinstall and tick "Add python.exe to PATH", or
use `py` instead of `python` everywhere.

**`'pip' is not recognized`**
Use `python -m pip` instead of `pip`.

**`running scripts is disabled on this system`** (PowerShell only)
See the note in step 4, or switch to Command Prompt.

**`Microsoft Visual C++ 14.0 or greater is required`**
pip is trying to compile a package from source because no wheel exists for
your Python version. Install Python 3.12 instead of a newer release.

**`Resource vader_lexicon not found`**
Re-run the download command at the end of step 5 with the venv activated.

**`SSL: CERTIFICATE_VERIFY_FAILED`**
Common on university or corporate networks that inspect traffic. Try a home
network or a phone hotspot.

**`MemoryError`, or the machine freezes during the full run**
Close other applications, or run on a sample:
`python pipeline.py --rows 100000`

**`Port 8501 is already in use`**
Another Streamlit is already running. Close it, or pick a different port:
`streamlit run app.py --server.port 8502`

**`FileNotFoundError: data/Reviews.csv`**
Step 7 was skipped. The dataset is too large for Git, so each machine needs
to download it.

**`No results yet. Run pipeline.py first.`** (in the dashboard)
`out\` is empty. Run step 8 before step 9.

---

## What gets installed where

| Item | Location | Size |
|---|---|---|
| Python | `C:\Users\<you>\AppData\Local\Programs\Python\Python312` | ~100 MB |
| Packages | `capstone\.venv` | ~500 MB |
| Dataset | `capstone\data` | ~410 MB |
| Results | `capstone\out` | ~45 MB |

To remove everything, delete the `capstone` folder and uninstall Python from
Settings > Apps.

# Windows setup from scratch

For a Windows machine with nothing installed — no Python, no Git. Follow in order.
Expect 30–45 minutes, most of it downloads.

**You need:** Windows 10 or 11, about 2 GB free disk, and an internet connection.
A Kaggle account is **not** required.

---

## 1. Install Python

Download **Python 3.12** from <https://www.python.org/downloads/windows/>
(choose "Windows installer (64-bit)").

> Use 3.12, not 3.13 or newer. Some packages don't publish Windows wheels for the
> newest versions yet, and pip then tries to compile them, which fails without a
> C++ compiler installed.

In the installer:

1. **Tick "Add python.exe to PATH"** at the bottom of the first screen. This is the
   step people miss, and skipping it causes the `'python' is not recognized` error.
2. Click **Install Now**.
3. On the last screen, if you see **"Disable path length limit"**, click it.

Check it worked. Open **Command Prompt** (press Start, type `cmd`, press Enter):

```
py --version
```

You should see `Python 3.12.x`. If you get an error, see
[Troubleshooting](#troubleshooting).

---

## 2. Install Git

Download from <https://git-scm.com/download/win> and run the installer. Accept every
default — there is nothing to change.

Skip this step if you are copying the project folder from a USB drive instead of
cloning it.

---

## 3. Get the project

In Command Prompt, pick a folder and clone:

```
cd %USERPROFILE%\Documents
git clone <your-repository-url> capstone
cd capstone
```

Replace `<your-repository-url>` with your GitHub URL.

**Copying from USB instead?** Copy the folder to `Documents\capstone`, then
`cd %USERPROFILE%\Documents\capstone`. Delete any `.venv` folder that came with it —
a virtual environment built on another machine will not run here.

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

This downloads about 150 MB and takes 3–10 minutes. Installing pandas,
scikit-learn, Streamlit, Plotly, NLTK and PyArrow.

Then fetch the sentiment dictionary VADER needs:

```
python -c "import nltk; nltk.download('vader_lexicon')"
```

---

## 6. Check it works, before downloading any data

```
python pipeline.py --test
```

Expected output: `self-check ok`

Now run the whole pipeline on the 600-row sample that ships with the project:

```
python pipeline.py --raw data/Reviews.sample.csv --rows 0 --topics 4
```

If that finishes and writes files into `out\`, your setup is correct and you can
move on to the real dataset.

---

## 7. Download the real dataset

```
python fetch_data.py
```

This downloads **122 MB** from Stanford SNAP and converts it to
`data\Reviews.csv` (289 MB, 568,454 reviews). It takes 5–15 minutes depending on
your connection.

No Kaggle account or API token is needed — SNAP is the original source that Kaggle
repackages.

---

## 8. Run the pipeline

Start with a sample to confirm everything holds together:

```
python pipeline.py --rows 50000
```

Then the full dataset:

```
python pipeline.py --rows 0 --topics 12
```

The full run takes **3–6 minutes** and needs about **4 GB of free RAM**. It prints
the ingestion funnel, the accuracy report, and the ranked opportunities, then writes
to `out\`.

---

## 9. Start the dashboard

```
streamlit run app.py
```

Your browser opens at <http://localhost:8501>. Leave the Command Prompt window open
while you use it — closing it stops the dashboard.

Press `Ctrl+C` in Command Prompt to stop.

---

## Coming back later

Every new Command Prompt session needs the environment activated first:

```
cd %USERPROFILE%\Documents\capstone
.venv\Scripts\activate
streamlit run app.py
```

Steps 1–7 are one-time only.

---

## Troubleshooting

**`'python' is not recognized as an internal or external command`**
Python isn't on PATH. Either reinstall and tick "Add python.exe to PATH", or use
`py` instead of `python` everywhere.

**`'pip' is not recognized`**
Use `python -m pip` instead of `pip`.

**`running scripts is disabled on this system`** (PowerShell only)
See the note in step 4, or just use Command Prompt.

**`Microsoft Visual C++ 14.0 or greater is required`**
pip is trying to compile a package from source because no wheel exists for your
Python version. Install Python 3.12 rather than a newer release.

**`Resource vader_lexicon not found`**
Re-run the download command at the end of step 5 with the environment activated.

**`SSL: CERTIFICATE_VERIFY_FAILED`**
Common on university or corporate networks that inspect traffic. Try a home network
or a phone hotspot.

**`MemoryError`, or the machine freezes during the full run**
Close other applications, or work with a sample instead:
`python pipeline.py --rows 100000`

**`Port 8501 is already in use`**
A dashboard is already running — check your other windows, or use a different port:
`streamlit run app.py --server.port 8502`

**`FileNotFoundError: data/Reviews.csv`**
You skipped step 7. The dataset is too large to store in Git, so every machine must
download it.

**`No results yet. Run pipeline.py first.`** (shown in the dashboard)
The `out\` folder is empty. Run step 8 before step 9.

---

## What gets installed where

| Item | Location | Size |
|---|---|---|
| Python | `C:\Users\<you>\AppData\Local\Programs\Python\Python312` | ~100 MB |
| Packages | `capstone\.venv` | ~500 MB |
| Dataset | `capstone\data` | ~410 MB |
| Results | `capstone\out` | ~45 MB |

To remove everything, delete the `capstone` folder and uninstall Python from
Settings → Apps.

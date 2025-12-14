# 🤖 Robotics Coursework 2 — robotics_cw2

This repository contains the codebase for **Robotics Coursework 2**.  
The project focuses on robot control, vision processing, and integration of multiple components into a single executable pipeline.

---
### Please run the main.py that is found in refactored/main.py rather than the one in the root folder.
---

## 🧑‍💻 Requirements

- Python 3.8 or newer
- pip

(Optional but recommended)
- Virtual environment (`venv`)

---

## ⚙️ Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate         # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

The **main entry point for this coursework** is:

```bash
python3 refactored/main.py
```

All execution, integration, and control logic should be run from this file.

---

## 📝 Notes

- `refactored/main.py` is the **only file required to run the full system**.
- Other scripts and folders support testing, training, or development.
- Vision-related functionality is implemented in `chess_vision.py`.
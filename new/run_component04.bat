@echo off
setlocal
cd /d "%~dp0"
echo Component 04 - PP1 Traffic Risk Demo
where py >nul 2>nul
if errorlevel 1 (
  echo Install Python 3.11 from python.org with Python Launcher enabled.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
  if errorlevel 1 goto failed
)
if not exist ".venv\dependencies_ready.txt" (
  .venv\Scripts\python.exe -m pip install --upgrade pip
  if errorlevel 1 goto failed
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 goto failed
  .venv\Scripts\python.exe -m pip freeze > .venv\installed_versions.txt
  echo ready> .venv\dependencies_ready.txt
)
.venv\Scripts\python.exe -m streamlit run app.py --server.address localhost
if errorlevel 1 goto failed
exit /b 0
:failed
echo Setup or launch failed. Read the error above. Python 3.11 and internet are needed for first setup.
echo To retry dependencies after a failure or requirements change, delete .venv\dependencies_ready.txt if present.
pause
exit /b 1

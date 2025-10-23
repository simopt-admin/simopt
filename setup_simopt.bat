@echo off
set ENV_NAME=simopt
set YML_FILE=environment.yml

echo Checking for Conda installation...
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Conda not found! Please install Miniconda or Anaconda first.
    exit /b 1
)

:: Ensure Conda is initialized
call "%USERPROFILE%\miniconda3\Scripts\activate.bat"

:: Check if environment already exists
conda env list | findstr /C:"%ENV_NAME%" >nul
if %errorlevel% == 0 (
    echo Environment "%ENV_NAME%" exists. Updating...
    call conda env update --name "%ENV_NAME%" --file "%YML_FILE%" --prune
) else (
    echo Creating new environment "%ENV_NAME%"...
    call conda env create -f "%YML_FILE%"
)

:: Activate environment
call conda activate %ENV_NAME%

echo Setup complete! Run: conda activate %ENV_NAME%

@echo off
cd /d C:\Users\senio\OneDrive\Desktop\repos\portfolio_dashboard
echo [%date% %time%] Starting journal.py >> logs\scheduler.log 2>&1
python jobs\journal.py >> logs\scheduler.log 2>&1
echo [%date% %time%] journal.py finished with exit code %errorlevel% >> logs\scheduler.log 2>&1

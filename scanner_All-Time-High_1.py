"""
All-Time-High email scanner (the Scanner_All-Time-Highs job runs this file).

All logic lives in ath_scanner_backtest.py so the email and the dashboard always use the
same definition of a confirmed ATH run and the same rules file (ath_scanner_rules.json).
Change the rules in the dashboard, commit the rules file, and this picks them up.
Running it directly is the same as:  python ath_scanner_backtest.py --alert
"""

from ath_scanner_backtest import run_alerts

if __name__ == "__main__":
    run_alerts()

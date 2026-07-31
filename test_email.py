"""
test_email.py
--------------
Standalone script to debug the Gmail SMTP setup outside of Streamlit,
so the real error message shows up instead of the app's generic
"Could not send the reset email" message.

Run locally with:
    python test_email.py

Edit the three values below before running (or set them as environment
variables GMAIL_ADDRESS / GMAIL_APP_PASSWORD / TEST_RECIPIENT).
"""

import os
import smtplib
from email.mime.text import MIMEText

GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "emnamallek29@gmail.com")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "tyjnslzclyjirwqk")
TEST_RECIPIENT = os.environ.get("TEST_RECIPIENT", GMAIL_ADDRESS)  # send to yourself by default

msg = MIMEText("This is a test email from test_email.py")
msg["Subject"] = "SMTP test"
msg["From"] = GMAIL_ADDRESS
msg["To"] = TEST_RECIPIENT

print(f"Connecting to smtp.gmail.com:587 as {GMAIL_ADDRESS} ...")

try:
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
        server.set_debuglevel(1)  # prints the full SMTP conversation
        server.starttls()
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, [TEST_RECIPIENT], msg.as_string())
    print("\n✅ SUCCESS — check the inbox of", TEST_RECIPIENT)
except Exception as e:
    print("\n❌ FAILED with error:")
    print(type(e).__name__, "-", e)
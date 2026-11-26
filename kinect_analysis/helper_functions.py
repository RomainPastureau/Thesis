import os
from datetime import datetime
import smtplib
from collections import defaultdict
from email.mime.text import MIMEText

from krajjat import tool_functions as ktools
from constants import *
import matplotlib.pyplot as plt

def get_delays():
    delays = defaultdict(dict)
    with open("delays.txt", "r") as f:
        for line in f:
            if line.strip():
                person, video, delay = line.strip().split("\t")
                delays[person][video] = float(delay)
    return delays

def get_conditions():
    table = ktools.read_xlsx(PATH_INFO, "Videos", read_metadata=False)[0]

    masked = {}

    for l in range(1, len(table)):
        if table[l][0] is not None:
            if table[l][0] not in masked.keys():
                masked[table[l][0]] = {}
            masked[table[l][0]][table[l][2].split(".")[0]] = table[l][1]

    return masked

def format_duration(since):
    delta = datetime.now() - since
    total_seconds = int(delta.total_seconds())
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days > 0:
        return f"{days}d {hours:02d}h{minutes:02d}m{seconds:02d}s"
    return f"{hours:02d}h{minutes:02d}m{seconds:02d}s"

def send_email(subject, body, to_email, from_email, password, smtp_server="127.0.0.1", smtp_port=1025):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
    except Exception as e:
        print(f"Failed to send email: {e}")

def save_figure(fig, directory, name, **kwargs):
    path = os.path.join(directory, name)
    fig.savefig(path, **kwargs)
    plt.close(fig)
    print(f"  Saved: {path}")

def print_section_title(title, length=60):
    print("\n" + "=" * length)
    print(title)
    print("=" * length)
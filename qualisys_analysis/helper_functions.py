import smtplib
from datetime import datetime
from email.mime.text import MIMEText
import matplotlib.pyplot as plt
import os
from constants import *

from krajjat import tool_functions as ktools

def get_info(subjects):

    excel_table, _ = ktools.read_xlsx(PATH_INFO, read_metadata=False)

    data = {subject: {} for subject in subjects}

    col_subject = excel_table[0].index("Participant")
    col_project = excel_table[0].index("Project")
    col_recording = excel_table[0].index("Recording")
    col_delay = excel_table[0].index("Delay from CCF (values)")
    col_delay_audio = excel_table[0].index("Delay audio (values)")
    col_retelling = excel_table[0].index("Retelling")
    col_cartoon = excel_table[0].index("Cartoon")
    col_has_gesture = excel_table[0].index("Gesture?")

    for row in excel_table:
        if row[col_subject] in subjects and row[col_retelling] not in ["Calibration", "Error"]:
            subject = row[col_subject]
            recording = row[col_retelling]
            if subject == "S021" and recording == "R001":
                row[col_delay] = None
            if row[col_delay] is None or row[col_delay_audio] is None:
                delay = None
            else:
                delay = row[col_delay] + row[col_delay_audio]

            data[subject][recording] = {"delay": delay,
                                        "cartoon": row[col_cartoon],
                                        "has gesture": row[col_has_gesture],
                                        "project": row[col_project],
                                        "video": row[col_recording]}

    return data

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
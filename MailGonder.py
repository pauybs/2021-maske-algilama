import smtplib

def MailGonder(mesaj):
    mailicerigi = mesaj
    mail = smtplib.SMTP("smtp.gmail.com",587)
    mail.ehlo()
    mail.starttls()
    mail.login("gönderenmail","gönderensifre")
    mail.sendmail("gönderenmail","alıcımail",mailicerigi)
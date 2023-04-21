package org.example;

import javax.mail.*;
import javax.mail.internet.InternetAddress;
import javax.mail.internet.MimeBodyPart;
import javax.mail.internet.MimeMessage;
import javax.mail.internet.MimeMultipart;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Properties;

public class SendEmailTLS {

    private String mailToSend, subjectToSend, messageToSend, filePath;
    private File file;
    private File passwordFile = new File("password.txt");
    private String password = new String(Files.readAllBytes(passwordFile.toPath()));
    private String myMail ="student35196ans@gmail.com", myPassword=password;

    public SendEmailTLS(String _mail, String _subject, String _message, String _filePath) throws IOException {
        mailToSend = _mail;
        subjectToSend = _subject;
        messageToSend = _message;
        filePath = _filePath;
        file = new File(filePath);
    }

    public void sendMessage() {
        final String username = myMail;
        final String password = myPassword;

        Properties prop = new Properties();
        prop.put("mail.smtp.host", "smtp.gmail.com");
        prop.put("mail.smtp.port", "587");
        prop.put("mail.smtp.auth", "true");
        prop.put("mail.smtp.starttls.enable", "true");

        Session session = Session.getInstance(prop,
                new javax.mail.Authenticator() {
                    protected PasswordAuthentication getPasswordAuthentication() {
                        return new PasswordAuthentication(username, password);
                    }
                });

        try {
            Message message = new MimeMessage(session);
            message.setFrom(new InternetAddress(myMail));
            message.setRecipients(
                    Message.RecipientType.TO,
                    InternetAddress.parse(mailToSend)
            );
            message.setSubject(subjectToSend);

            Multipart multipart = new MimeMultipart();
            MimeBodyPart messageBodyPart = new MimeBodyPart();
            messageBodyPart.setText(messageToSend);
            multipart.addBodyPart(messageBodyPart);

            if(!filePath.isEmpty()){
                MimeBodyPart imagePart = new MimeBodyPart();
                javax.activation.DataSource source = new javax.activation.FileDataSource(filePath);
                imagePart.setDataHandler(new javax.activation.DataHandler(source));
                imagePart.setFileName(file.getName());
                multipart.addBodyPart(imagePart);
            }
            message.setContent(multipart);

            Transport.send(message);
            System.out.println("Done");

        } catch (MessagingException e) {
            e.printStackTrace();
        }
    }
}
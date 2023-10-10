package Mail;

import javax.mail.*;
import javax.mail.internet.InternetAddress;
import javax.mail.internet.MimeMessage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Properties;

/**
 * This class provides functionality to send an email using TLS (Transport Layer Security) protocol.
 * It uses the JavaMail API to connect to an SMTP server and send the email.
 */
public class SendEmailTLS {
    private String mailToSend, subjectToSend, messageToSend;
    private File passwordFile = new File("password.txt");
    private String password = new String(Files.readAllBytes(passwordFile.toPath()));
    private String myMail ="fiszkiapp@gmail.com", myPassword=password;

    /**
     * Constructs a new SendEmailTLS object with the specified email details.
     *
     * @param _mail    the email address to send the email to
     * @param _subject the subject of the email
     * @param _message the content of the email
     * @throws IOException if an I/O error occurs while reading the password from file
     */
    public SendEmailTLS(String _mail, String _subject, String _message) throws IOException {
        mailToSend = _mail;
        subjectToSend = _subject;
        messageToSend = _message;
    }

    /**
     * Sends the email using the provided email details.
     */
    public void sendMessage() {
        final String username = myMail;
        final String password = myPassword;

        Properties prop = new Properties();
        prop.put("mail.smtp.host", "smtp.gmail.com");
        prop.put("mail.smtp.port", "587");
        prop.put("mail.smtp.auth", "true");
        prop.put("mail.smtp.starttls.enable", "true"); //TLS

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
            message.setText(messageToSend);
            Transport.send(message);
            //System.out.println("Done");

        } catch (MessagingException e) {
            e.printStackTrace();
        }
    }
}
package com.holiday.request.service;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class MailService {

    private final JavaMailSender emailSender;



    @Value("${spring.mail.username}")
    private String sender;

    public void sendEmail(String toEmail, String subject, String body) {
        SimpleMailMessage message = new SimpleMailMessage();

        message.setFrom(sender);
        message.setTo(toEmail);
        message.setSubject(subject);
        message.setText(body);
        try {
            emailSender.send(message);

        } catch(Exception e) {
            System.out.println(e);
        }
    }
}

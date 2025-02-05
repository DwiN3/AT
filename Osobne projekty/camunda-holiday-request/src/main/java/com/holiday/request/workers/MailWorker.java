package com.holiday.request.workers;

import com.holiday.request.service.MailService;
import io.camunda.zeebe.client.api.response.ActivatedJob;
import io.camunda.zeebe.spring.client.annotation.JobWorker;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Objects;

@Component
@RequiredArgsConstructor
public class MailWorker {
    private final MailService mailService;

    private final int APPROVED = 1;
    private final int DECLINED = 0;
    private final String SUBJECT_DECLINED = "Odmowa urlopu";
    private final String SUBJECT_APPROVED= "Potwierdzenie urlopu";
    private final String BODY_DECLINED = "Witam, niestety nie możemy zatwierdzić Twojego urlopu w tym terminie.";
    private final String BODY_APPROVED = "Witam, twój urlop został zaakceptowany";

    @JobWorker(type = "send_mail")
    public void sendMail(final ActivatedJob job) {

        Map<String, Object> jobVariables = job.getVariablesAsMap();
        String toEmail = (String) jobVariables.get("email");
        var isApproved = (int) jobVariables.get("type");
        if (Objects.equals(isApproved, APPROVED))
            mailService.sendEmail(toEmail, SUBJECT_APPROVED, BODY_APPROVED);
        else
            mailService.sendEmail(toEmail, SUBJECT_DECLINED, BODY_DECLINED);

        System.out.println("E-mail wysłany do: " + toEmail);
    }
}

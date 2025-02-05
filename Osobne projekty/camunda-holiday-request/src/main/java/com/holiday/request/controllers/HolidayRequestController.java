package com.holiday.request.controllers;

import io.camunda.zeebe.client.ZeebeClient;
import lombok.AllArgsConstructor;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/api/v1")
@AllArgsConstructor
public class HolidayRequestController {

    private static final String BPMN_PROCESS_ID = "Process_02wsd0v";


    @Qualifier("zeebeClientLifecycle")
    private ZeebeClient client;

    @PostMapping("/start")
    public Map<String, Object> startProcessInstance(@RequestBody Map<String, Object> variables) {
        var event = client
                .newCreateInstanceCommand()
                .bpmnProcessId(BPMN_PROCESS_ID)
                .latestVersion()
                .variables(variables)
                .send();

        variables.put("processInstanceKey", event.join().getProcessInstanceKey());
        variables.put("bpmnProcessId", BPMN_PROCESS_ID);

        System.out.println("Process instance key: " + event.join().getProcessInstanceKey());
        return variables;
    }





}

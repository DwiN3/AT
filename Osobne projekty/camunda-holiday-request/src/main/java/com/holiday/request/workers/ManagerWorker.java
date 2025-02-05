package com.holiday.request.workers;


import com.holiday.request.service.LeaveRequestService;
import io.camunda.zeebe.client.api.response.ActivatedJob;
import io.camunda.zeebe.spring.client.annotation.JobWorker;
import lombok.AllArgsConstructor;
import org.springframework.stereotype.Component;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.Map;

@Component
@AllArgsConstructor
public class ManagerWorker {

    private final LeaveRequestService leaveRequestService;

    @JobWorker(type = "consideration")
    public Map<String, Object> consideration(final ActivatedJob job) {

        var jobResultVariables = job.getVariablesAsMap();
        String startDate = (String) jobResultVariables.get("startDate");
        String endDate = (String) jobResultVariables.get("endDate");

        final DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd");
        LocalDate date = LocalDate.parse(startDate, dtf);
        LocalDate date2 = LocalDate.parse(endDate, dtf);

        var isAvailable = leaveRequestService.checkLeaveRequest(date, date2);
        if (isAvailable) {
            jobResultVariables.put("decision", 1);
        }
        else {
            var dates = leaveRequestService.findNextAvailableDates(date, date2);
            if (dates.isEmpty()){
                jobResultVariables.put("decision", 0);

            } else {
                jobResultVariables.put("decision", 2);
                jobResultVariables.put("availableDates", dates);
            }
        }
        return jobResultVariables;

    }

}

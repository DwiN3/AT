package com.holiday.request.dto.request;

import com.holiday.request.enums.LeaveStatus;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDate;
import java.util.Date;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CreateLeaveRequestDTO {
    private int employeeId;
    private LocalDate  startDate;
    private LocalDate endDate;
}
package com.holiday.request.dto.model;

import com.holiday.request.enums.LeaveStatus;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.Date;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class LeaveRequestDTO {
    private int id;
    private int employeeId;
    private Date startDate;
    private Date endDate;
    private LeaveStatus status;
}

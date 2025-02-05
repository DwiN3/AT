package com.holiday.request.dto.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class EmployeeDTO {
    private int id;
    private String name;
    private String email;
    private List<LeaveRequestDTO> leaveRequests;
}

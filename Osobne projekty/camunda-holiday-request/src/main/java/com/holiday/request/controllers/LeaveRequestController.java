package com.holiday.request.controllers;

import com.holiday.request.dto.model.LeaveRequestDTO;
import com.holiday.request.dto.request.CreateLeaveRequestDTO;
import com.holiday.request.model.LeaveRequest;
import com.holiday.request.service.LeaveRequestService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/leave-request")
public class LeaveRequestController {

    private final LeaveRequestService leaveRequestService;

    @PostMapping("/create")
    public ResponseEntity<?> addLeaveRequest(@RequestBody CreateLeaveRequestDTO requestDTO) {
        LeaveRequest createdLeaveRequest = leaveRequestService.create(requestDTO);
        return ResponseEntity.ok(createdLeaveRequest);
    }

    @GetMapping("/all")
    public ResponseEntity<List<LeaveRequestDTO>> getAllLeaveRequests() {
        List<LeaveRequestDTO> leaveRequests = leaveRequestService.getAllLeaveRequests();
        return ResponseEntity.ok(leaveRequests);
    }


    @GetMapping("/by-status")
    public ResponseEntity<List<LeaveRequestDTO>> getLeaveRequestsByStatus(@RequestParam String status) {
        List<LeaveRequestDTO> leaveRequests = leaveRequestService.getLeaveRequestsByStatus(status);
        return ResponseEntity.ok(leaveRequests);
    }

    @PatchMapping("/update-status/{id}")
    public ResponseEntity<LeaveRequestDTO> updateLeaveRequestStatus(@PathVariable int id, @RequestParam String status) {
        LeaveRequestDTO updatedLeaveRequest = leaveRequestService.updateLeaveRequestStatus(id, status);
        return ResponseEntity.ok(updatedLeaveRequest);
    }

    @GetMapping("/check-leave-request")
    public ResponseEntity<Boolean> checkLeaveRequest(@RequestParam("startDate") String startDate, @RequestParam("endDate") String endDate) {
        LocalDate start = LocalDate.parse(startDate);
        LocalDate end = LocalDate.parse(endDate);
        boolean isAvailable = leaveRequestService.checkLeaveRequest(start, end);
        return ResponseEntity.ok(isAvailable);
    }

    @GetMapping("/next-available-dates")
    public ResponseEntity<List<String>> getNextAvailableDates(@RequestParam("startDate") String startDate, @RequestParam("endDate") String endDate) {
        LocalDate start = LocalDate.parse(startDate);
        LocalDate end = LocalDate.parse(endDate);

        List<String> availableDates = leaveRequestService.findNextAvailableDates(start, end);
        return ResponseEntity.ok(availableDates);
    }

    @PatchMapping("/change-dates-and-approve/{id}")
    public ResponseEntity<LeaveRequestDTO> changeDatesAndApprove(@PathVariable int id, @RequestParam String startDate, @RequestParam String endDate) {
        LocalDate newStartDate = LocalDate.parse(startDate);
        LocalDate newEndDate = LocalDate.parse(endDate);
        LeaveRequestDTO updatedLeaveRequest = leaveRequestService.changeDatesAndApprove(id, newStartDate, newEndDate);
        return ResponseEntity.ok(updatedLeaveRequest);
    }
}

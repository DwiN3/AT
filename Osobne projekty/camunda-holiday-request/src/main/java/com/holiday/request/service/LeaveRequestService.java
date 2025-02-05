package com.holiday.request.service;

import com.holiday.request.dto.model.DayDto;
import com.holiday.request.dto.model.LeaveRequestDTO;
import com.holiday.request.dto.request.CreateLeaveRequestDTO;
import com.holiday.request.enums.LeaveStatus;
import com.holiday.request.model.Day;
import com.holiday.request.model.Employee;
import com.holiday.request.model.LeaveRequest;
import com.holiday.request.repository.DayRepository;
import com.holiday.request.repository.EmployeeRepository;
import com.holiday.request.repository.LeaveRequestRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;


@Service
@RequiredArgsConstructor
public class LeaveRequestService {

    private final LeaveRequestRepository leaveRequestRepository;
    private final EmployeeRepository employeeRepository;
    private final DayRepository dayRepository;

    public LeaveRequest create(CreateLeaveRequestDTO requestDTO) {
        Employee employee = employeeRepository.findById(requestDTO.getEmployeeId())
                .orElseThrow(() -> new IllegalArgumentException("Employee not found"));

        if (requestDTO.getEndDate().isBefore(requestDTO.getStartDate())) {
            throw new IllegalArgumentException("End date cannot be before start date");
        }

        Date startDate = Date.from(requestDTO.getStartDate().atStartOfDay(ZoneId.of("UTC")).toInstant());
        Date endDate = Date.from(requestDTO.getEndDate().atTime(23, 59, 59).atZone(ZoneId.of("UTC")).toInstant());

        LeaveRequest leaveRequest = LeaveRequest.builder()
                .employee(employee)
                .startDate(startDate)
                .endDate(endDate)
                .status(LeaveStatus.PENDING)
                .build();

        return leaveRequestRepository.save(leaveRequest);
    }

    public List<LeaveRequestDTO> getAllLeaveRequests() {
        return leaveRequestRepository.findAll().stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    public List<LeaveRequestDTO> getLeaveRequestsByStatus(String status) {
        LeaveStatus leaveStatus;
        try {
            leaveStatus = LeaveStatus.valueOf(status.toUpperCase());
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Invalid status: " + status);
        }

        return leaveRequestRepository.findAll().stream()
                .filter(leaveRequest -> leaveRequest.getStatus() == leaveStatus)
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    public LeaveRequestDTO updateLeaveRequestStatus(int id, String status) {
        LeaveRequest leaveRequest = leaveRequestRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Leave request not found with ID: " + id));

        LeaveStatus leaveStatus;
        try {
            leaveStatus = LeaveStatus.valueOf(status.toUpperCase());
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Invalid status: " + status);
        }

        if (leaveRequest.getStatus() != leaveStatus) {
            LocalDate startDate = leaveRequest.getStartDate().toInstant().atZone(ZoneId.systemDefault()).toLocalDate();
            LocalDate endDate = leaveRequest.getEndDate().toInstant().atZone(ZoneId.systemDefault()).toLocalDate();

            Date rangeStartDate = Date.from(startDate.atStartOfDay(ZoneId.systemDefault()).toInstant());
            Date rangeEndDate = Date.from(endDate.atStartOfDay(ZoneId.systemDefault()).toInstant());

            List<Day> daysInRange = dayRepository.findByDateBetween(rangeStartDate, rangeEndDate);

            if (leaveStatus == LeaveStatus.APPROVED) {
                daysInRange.forEach(day -> day.setAvailable(false));
            } else if (leaveStatus == LeaveStatus.REJECTED) {
                daysInRange.forEach(day -> day.setAvailable(true));
            }

            dayRepository.saveAll(daysInRange);
        }

        leaveRequest.setStatus(leaveStatus);
        LeaveRequest updatedLeaveRequest = leaveRequestRepository.save(leaveRequest);

        return convertToDTO(updatedLeaveRequest);
    }

    public boolean checkLeaveRequest(LocalDate startDate, LocalDate endDate) {

        if (endDate.isBefore(startDate)) {
            throw new IllegalArgumentException("End date cannot be before start date");
        }

        List<Day> daysInRange = dayRepository.findByDateBetween(
                Date.from(startDate.atStartOfDay(ZoneId.systemDefault()).toInstant()),
                Date.from(endDate.atStartOfDay(ZoneId.systemDefault()).toInstant())
        );

        return daysInRange.stream().allMatch(Day::isAvailable);
    }

    public List<String> findNextAvailableDates(LocalDate startDate, LocalDate endDate) {
        int daysInRange = (int) (endDate.toEpochDay() - startDate.toEpochDay() + 1);
        List<Day> allDays = dayRepository.findAllByOrderByDateAsc();

        List<Day> futureDays = allDays.stream()
                .filter(day -> !day.getDate().toInstant().atZone(ZoneId.systemDefault()).toLocalDate().isBefore(startDate))
                .collect(Collectors.toList());

        List<String> result = new ArrayList<>();
        int startIndex = 0;

        List<DayDto> dayDtos = futureDays.stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());

        while (result.size() < 3 && startIndex < dayDtos.size()) {
            List<DayDto> subList = dayDtos.stream()
                    .skip(startIndex)
                    .limit(daysInRange)
                    .collect(Collectors.toList());

            boolean allAvailable = subList.size() == daysInRange && subList.stream().allMatch(DayDto::isAvailable);

            if (allAvailable) {
                LocalDate rangeStart = subList.get(0).getDate().toInstant().atZone(ZoneId.systemDefault()).toLocalDate();
                LocalDate rangeEnd = subList.get(subList.size() - 1).getDate().toInstant().atZone(ZoneId.systemDefault()).toLocalDate();

                result.add(rangeStart + " - " + rangeEnd);

                startIndex += daysInRange;
            } else {
                startIndex++;
            }
        }

        return result;
    }

    public LeaveRequestDTO changeDatesAndApprove(int id, LocalDate newStartDate, LocalDate newEndDate) {
        LeaveRequest leaveRequest = leaveRequestRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Leave request not found with ID: " + id));

        if (newEndDate.isBefore(newStartDate)) {
            throw new IllegalArgumentException("End date cannot be before start date");
        }

        Date startDate = Date.from(newStartDate.atStartOfDay(ZoneId.systemDefault()).toInstant());
        Date endDate = Date.from(newEndDate.atStartOfDay(ZoneId.systemDefault()).toInstant());

        leaveRequest.setStartDate(startDate);
        leaveRequest.setEndDate(endDate);
        leaveRequest.setStatus(LeaveStatus.APPROVED);

        LeaveRequest updatedLeaveRequest = leaveRequestRepository.save(leaveRequest);
        List<Day> daysInRange = dayRepository.findByDateBetween(startDate, endDate);

        daysInRange.forEach(day -> day.setAvailable(false));
        dayRepository.saveAll(daysInRange);

        return convertToDTO(updatedLeaveRequest);
    }

    private LeaveRequestDTO convertToDTO(LeaveRequest leaveRequest) {
        return LeaveRequestDTO.builder()
                .id(leaveRequest.getId())
                .employeeId(leaveRequest.getEmployee().getId())
                .startDate(leaveRequest.getStartDate())
                .endDate(leaveRequest.getEndDate())
                .status(leaveRequest.getStatus())
                .build();
    }

    private DayDto convertToDTO(Day day) {
        return DayDto.builder()
                .id(day.getId())
                .date(day.getDate())
                .available(day.isAvailable())
                .build();
    }
}
package com.holiday.request.service;

import com.holiday.request.dto.model.EmployeeDTO;
import com.holiday.request.dto.request.CreateEmployeeDTO;
import com.holiday.request.model.Employee;
import com.holiday.request.repository.EmployeeRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class EmployeeService {

    private final EmployeeRepository employeeRepository;
    private final MailService mailService;

    public EmployeeDTO addEmployee(CreateEmployeeDTO employeeDTO) {
        Employee employee = Employee.builder()
                .name(employeeDTO.getName())
                .email(employeeDTO.getEmail())
                .build();
        Employee savedEmployee = employeeRepository.save(employee);
        return convertToDTO(savedEmployee);
    }

    public void deleteEmployee(int id) {
        if (!employeeRepository.existsById(id)) {
            throw new IllegalArgumentException("Employee with ID " + id + " does not exist.");
        }
        employeeRepository.deleteById(id);
    }

    public List<EmployeeDTO> getAllEmployees() {
        return employeeRepository.findAll().stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    private EmployeeDTO convertToDTO(Employee employee) {
        return EmployeeDTO.builder()
                .id(employee.getId())
                .name(employee.getName())
                .email(employee.getEmail())
                .build();
    }
    public void sendEmail(String email) {
        mailService.sendEmail(email, "subject", "body");
    }
}
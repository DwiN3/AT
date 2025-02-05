package com.holiday.request.controllers;

import com.holiday.request.dto.model.EmployeeDTO;
import com.holiday.request.dto.request.CreateEmployeeDTO;
import com.holiday.request.model.Employee;
import com.holiday.request.service.EmployeeService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/employee")
public class EmployeeController {

    private final EmployeeService employeeService;

    @PostMapping("/add")
    public ResponseEntity<EmployeeDTO> addEmployee(@RequestBody CreateEmployeeDTO employeeDTO) {
        EmployeeDTO createdEmployee = employeeService.addEmployee(employeeDTO);
        return ResponseEntity.ok(createdEmployee);
    }

    @DeleteMapping("/delete/{id}")
    public ResponseEntity<String> deleteEmployee(@PathVariable int id) {
        employeeService.deleteEmployee(id);
        return ResponseEntity.ok("Employee with ID " + id + " deleted successfully.");
    }

    @GetMapping("/all")
    public ResponseEntity<List<EmployeeDTO>> getAllEmployees() {
        List<EmployeeDTO> employees = employeeService.getAllEmployees();
        return ResponseEntity.ok(employees);
    }

    @PostMapping("/{email}")
    public void SendEmail(@PathVariable String email) {
        employeeService.sendEmail(email);
    }
}
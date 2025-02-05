package com.holiday.request.repository;

import com.holiday.request.dto.model.DayDto;
import com.holiday.request.model.Day;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.Date;
import java.util.List;

@Repository
public interface DayRepository extends JpaRepository<Day, Integer> {
    @Query("SELECT d FROM Day d WHERE d.date >= :startDate AND d.date <= :endDate")
    List<Day> findByDateBetween(@Param("startDate") Date startDate, @Param("endDate") Date endDate);
    List<Day> findAllByOrderByDateAsc();
}
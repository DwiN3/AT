package com.example.Fiszki;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Main class for the Fiszki application.
 * This class serves as the entry point for the Spring Boot application.
 *
 * @author Dereń, Polak, Kubik
 * @version 1.0
 * @since 2024-01-15
 */
@SpringBootApplication
public class FiszkiApplication {
	/**
	 * The main method that starts the Spring Boot application.
	 *
	 * @param args Command-line arguments passed to the application.
	 */
	public static void main(String[] args) {
		SpringApplication.run(FiszkiApplication.class, args);
	}
}
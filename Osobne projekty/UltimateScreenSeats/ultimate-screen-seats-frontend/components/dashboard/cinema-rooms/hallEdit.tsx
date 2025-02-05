"use client";

import React, { useState, useEffect } from "react";
import { Modal, ModalBody, ModalContent, ModalFooter, ModalHeader, ModalProps } from "@heroui/modal";
import { Button } from "@nextui-org/button";
import { Input } from "@nextui-org/input";

import { showToast } from "@/lib/showToast";
import { useAuth } from "@/providers/authProvider";

interface Hall {
  id: number;
  name: string;
  seat_layout: number[][];
  number_of_seats: number;
}

const HALLS_URL = "api/halls";

interface EditHallModalProps {
  isOpen: boolean;
  onClose: () => void;
  hall?: Hall;
  onSave: (updatedHall: Hall) => void;
}

export default function EditHallModal({
  isOpen,
  onClose,
  hall,
  onSave,
}: EditHallModalProps) {
  const [formData, setFormData] = useState<Hall>(
    hall || {
      id: Date.now(),
      name: "",
      seat_layout: [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ],
      number_of_seats: 0,
    }
  );
  const [errors, setErrors] = useState<Record<string, boolean>>({});
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const auth = useAuth();

  useEffect(() => {
    const totalSeats = formData.seat_layout.flat().filter((seat) => seat === 0).length;

    setFormData((prev) => ({ ...prev, number_of_seats: totalSeats }));
  }, [formData.seat_layout]);

  const validateForm = (): boolean => {
    const newErrors: Record<string, boolean> = {};
    let hasErrors = false;

    if (!formData.name) {
      newErrors.name = true;
      hasErrors = true;
    }

    setErrors(newErrors);
    if (hasErrors) {
      setErrorMessage("Wypełnij wszystkie wymagane pola.");
    } else {
      setErrorMessage(null);
    }

    return !hasErrors;
  };

  const handleChange = (key: keyof Hall, value: any) => {
    setFormData((prev) => ({ ...prev, [key]: value }));
  };

  const handleSeatClick = (rowIndex: number, colIndex: number) => {
    const updatedLayout = formData.seat_layout.map((row, rIndex) =>
      row.map((seat, cIndex) => {
        if (rIndex === rowIndex && cIndex === colIndex) {
          return seat === 0 ? -1 : 0;
        }

        return seat;
      })
    );

    setFormData((prev) => ({ ...prev, seat_layout: updatedLayout }));
  };

  const handleAddRow = () => {
    setFormData((prev) => ({
      ...prev,
      seat_layout: [...prev.seat_layout, Array(prev.seat_layout[0].length).fill(0)],
    }));
  };

  const handleRemoveRow = () => {
    if (formData.seat_layout.length > 1) {
      setFormData((prev) => ({
        ...prev,
        seat_layout: prev.seat_layout.slice(0, -1),
      }));
    }
  };

  const handleAddColumn = () => {
    setFormData((prev) => ({
      ...prev,
      seat_layout: prev.seat_layout.map((row) => [...row, 0]),
    }));
  };

  const handleRemoveColumn = () => {
    if (formData.seat_layout[0].length > 1) {
      setFormData((prev) => ({
        ...prev,
        seat_layout: prev.seat_layout.map((row) => row.slice(0, -1)),
      }));
    }
  };

  const handleSubmit = async () => {
    if (!validateForm()) {
      return;
    }

    try {
        const method = hall ? "PATCH" : "POST";
        const url = hall ? `${HALLS_URL}/${hall.id}` : HALLS_URL;
  
        const response = await fetch(url, {
          method,
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(formData)
        });
  
        if (response.status === 401) {
          auth.loginRequired();
  
          return;
        }
  
        if (!response.ok) {
          throw new Error(hall ? "Nie udało się zaktualizować sali kinowej!" : "Nie udało sie dodać sali kinowej!");
        }
  
        showToast(hall ? "Sala kinowa zaktualizowana pomyślnie!" : "Sala kinowa utworzona pomyślnie!", false);
  
        onSave(formData);
        onClose();
      } catch {
        showToast("Wystąpił błąd podczas zapisywania sali kinowej.", true);
      }
  };

  const [scrollBehavior] = React.useState<ModalProps["scrollBehavior"]>("inside");

  return (
    <Modal
      className="py-2"
      isDismissable={false}
      isOpen={isOpen}
      scrollBehavior={scrollBehavior}
      size="2xl"
      onOpenChange={(isOpen) => !isOpen && onClose()}
    >
      <ModalContent>
        {() => (
          <>
            <ModalHeader className="text-2xl font-bold text-primary">
              {hall ? `Edytuj salę ${hall.name}` : "Dodaj nową salę"}
            </ModalHeader>
            <ModalBody>
              {errorMessage && (
                <div className="text-danger font-bold text-lg text-center mb-4">
                  {errorMessage}
                </div>
              )}
              <Input
                errorMessage={errors.name ? "To pole jest wymagane." : undefined}
                isInvalid={errors.name}
                label="Nazwa sali"
                value={formData.name}
                onChange={(e) => handleChange("name", e.target.value)}
              />
              <div className="mt-4">
                <p className="font-bold mb-2">Układ siedzeń</p>
                <div className="inline-grid gap-1 border p-3 rounded-md bg-gray-50">
                  {formData.seat_layout.map((row, rowIndex) => (
                    <div key={rowIndex} className="flex">
                        {row.map((seat, colIndex) => (
                        <div
                            key={`${rowIndex}-${colIndex}`}
                            aria-label={`Seat ${rowIndex + 1}, ${colIndex + 1}`}
                            className={`w-8 h-8 flex items-center justify-center rounded cursor-pointer ${
                            seat === -1
                                ? "bg-gray-400"
                                : "bg-primary-500 text-white hover:bg-primary-700"
                            }`}
                            role="button"
                            tabIndex={0}
                            onClick={() => handleSeatClick(rowIndex, colIndex)}
                            onKeyDown={(e) => {
                            if (e.key === "Enter" || e.key === " ") {
                                e.preventDefault();
                                handleSeatClick(rowIndex, colIndex);
                            }
                            }}
                        >
                            {seat === 0 ? "S" : ""}
                        </div>
                        ))}
                    </div>
                  ))}
                </div>
                <div className="py-4">
                  <div className="flex items-center mb-2">
                      <div className="flex items-center justify-center rounded cursor-pointer bg-primary-500 text-white hover:bg-primary-700 h-8 w-8">S</div>
                      <p>&nbsp;- Miejsce siedzące</p>
                  </div>
                  <div className="flex items-center">
                      <div className="flex items-center justify-center rounded cursor-pointer bg-gray-400 h-8 w-8"/>
                      <p>&nbsp;- Brak miejsca siedzącego</p>
                  </div>
                </div>
                <div className="flex justify-between mt-4">
                  <Button color="success" size="sm" variant="flat" onPress={handleAddRow}>
                    Dodaj wiersz
                  </Button>
                  <Button color="danger" size="sm" variant="flat" onPress={handleRemoveRow}>
                    Usuń wiersz
                  </Button>
                  <Button color="success" size="sm" variant="flat" onPress={handleAddColumn}>
                    Dodaj kolumnę
                  </Button>
                  <Button color="danger" size="sm" variant="flat" onPress={handleRemoveColumn}>
                    Usuń kolumnę
                  </Button>
                </div>
              </div>
              <p className="mt-4">
                <strong>Liczba miejsc:</strong> {formData.number_of_seats}
              </p>
            </ModalBody>
            <ModalFooter>
              <Button color="danger" variant="ghost" onPress={onClose}>
                Anuluj
              </Button>
              <Button color="primary" variant="ghost" onPress={handleSubmit}>
                Zapisz
              </Button>
            </ModalFooter>
          </>
        )}
      </ModalContent>
    </Modal>
  );
}
"use client";

import React, { useState } from "react";
import { Modal, ModalBody, ModalContent, ModalFooter, ModalHeader } from "@heroui/modal";
import { Button } from "@nextui-org/button";
import { Input } from "@nextui-org/input";
import { Select, SelectItem } from "@nextui-org/select";

import { showToast } from "@/lib/showToast";

const SHOWINGS_URL = '/api/showings'


interface AddShowingModalProps {
  isOpen: boolean;
  onClose: () => void;
  movies: Movie[];
  cinemaRooms: CinemaRoom[];
  onSave: () => void;
}

export default function AddShowingModal({
  isOpen,
  onClose,
  movies,
  cinemaRooms,
  onSave,
}: AddShowingModalProps) {
  const [formData, setFormData] = useState({
    movie_id: "",
    cinema_room_id: "",
    date: "",
    ticket_price: "",
  });

  const [selectedCinemaRoom, setSelectedCinemaRoom] = useState<CinemaRoom | null>(null);
  const [errors, setErrors] = useState<Record<string, boolean>>({});

  const validateForm = (): boolean => {
    const newErrors: Record<string, boolean> = {};
    let hasErrors = false;

    if (!formData.movie_id) {
      newErrors.movie_id = true;
      hasErrors = true;
    }
    if (!formData.cinema_room_id) {
      newErrors.cinema_room_id = true;
      hasErrors = true;
    }
    if (!formData.date) {
      newErrors.date = true;
      hasErrors = true;
    }
    if (!formData.ticket_price || parseFloat(formData.ticket_price) <= 0) {
      newErrors.ticket_price = true;
      hasErrors = true;
    }

    setErrors(newErrors);

    return !hasErrors;
  };

  const handleSubmit = async () => {
    if (!validateForm()) {
      showToast("Wypełnij wszystkie pola!", true);

      return;
    }

    try {
      const response = await fetch(SHOWINGS_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error("Błąd podczas dodawania seansu.");
      }

      showToast("Seans dodany pomyślnie!", false);
      onSave();
      onClose();
    } catch {
      showToast("Nie udało się dodać seansu. ", true);
    }
  };

  const handleChange = (key: keyof typeof formData, value: string) => {
    setFormData((prev) => ({ ...prev, [key]: value }));
  };

  const handleCinemaRoomChange = (cinemaRoomId: string) => {
    const selectedRoom = cinemaRooms.find((room) => room.id.toString() === cinemaRoomId);

    setSelectedCinemaRoom(selectedRoom || null);
    handleChange("cinema_room_id", cinemaRoomId);
  };

  return (
    <Modal className="py-2" isDismissable={false} isOpen={isOpen} size="2xl" onOpenChange={(isOpen) => !isOpen && onClose()}>
      <ModalContent>
        {() => (
          <>
            <ModalHeader className="text-2xl font-bold text-primary">Dodaj Seans</ModalHeader>
            <ModalBody>
            <Select
                className=" mb-2"
                errorMessage={errors.movie_id ? "Wybierz film." : undefined}
                isInvalid={errors.movie_id}
                label="Film"
                labelPlacement="outside"
                placeholder="Wybierz film"
                selectedKeys={new Set([formData.movie_id])}
                onSelectionChange={(selected) => {
                    const selectedId = Array.from(selected)[0]?.toString();

                    if (selectedId) {
                        handleChange("movie_id", selectedId);
                    }
                }}
                >
                {movies.map((movie) => (
                    <SelectItem
                        key={movie.id.toString()}
                        value={movie.id.toString()}
                    >
                    {movie.title}
                    </SelectItem>
                ))}
            </Select>

              <Select
                className="mb-2"
                errorMessage={errors.cinema_room_id ? "Wybierz salę." : undefined}
                isInvalid={errors.cinema_room_id}
                label="Sala kinowa"
                labelPlacement="outside"
                placeholder="Wybierz salę"
                selectedKeys={new Set([formData.cinema_room_id])}
                onSelectionChange={(selected) =>
                  handleCinemaRoomChange(Array.from(selected)[0].toString())
                }
              >
                {cinemaRooms.map((room) => (
                  <SelectItem key={room.id.toString()} value={room.id.toString()}>
                    {room.name}
                  </SelectItem>
                ))}
              </Select>

              {selectedCinemaRoom && (
                <div className="my-2 ml-2">
                  <p className="text-md font-semibold mb-2">Układ Siedzeń</p>
                  <div
                    className="inline-grid gap-1"
                    style={{
                      gridTemplateColumns: `repeat(${selectedCinemaRoom.seat_layout[0]?.length || 1}, 1fr)`,
                    }}
                  >
                    {selectedCinemaRoom.seat_layout.flat().map((seat, index) => (
                      <div
                        key={index}
                        className={`w-6 h-6 flex items-center justify-center rounded text-xs ${
                          seat === -1 ? "bg-gray-500" : "bg-primary-400"
                        }`}
                      />
                    ))}
                  </div>
                </div>
              )}

              <Input
                className="mb-2"
                errorMessage={errors.date ? "Podaj datę seansu." : undefined}
                isInvalid={errors.date}
                label="Data seansu"
                labelPlacement="outside"
                placeholder="Podaj datę seansu"
                type="datetime-local"
                value={formData.date}
                onChange={(e) => handleChange("date", e.target.value)}
              />

              <Input
                className="mb-4"
                endContent={
                    <div className="pointer-events-none flex items-center">
                      <span className="text-default-400 text-small">zł</span>
                    </div>
                  }
                errorMessage={errors.ticket_price ? "Podaj cenę biletu." : undefined}
                isInvalid={errors.ticket_price}
                label="Cena biletu"
                labelPlacement="outside"
                placeholder="Podaj cenę biletu"
                type="number"
                value={formData.ticket_price}
                onChange={(e) => handleChange("ticket_price", e.target.value)}
              />
            </ModalBody>
            <ModalFooter>
              <Button color="danger" variant="ghost" onPress={onClose}>
                Anuluj
              </Button>
              <Button color="primary" variant="ghost" onPress={handleSubmit}>
                Dodaj Seans
              </Button>
            </ModalFooter>
          </>
        )}
      </ModalContent>
    </Modal>
  );
}

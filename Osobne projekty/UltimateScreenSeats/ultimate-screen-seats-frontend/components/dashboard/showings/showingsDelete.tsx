"use client";

import React, { useState } from "react";
import { Button } from "@nextui-org/button";
import { Modal, ModalBody, ModalContent, ModalFooter, ModalHeader } from "@heroui/modal";

import { useAuth } from "@/providers/authProvider";
import { showToast } from "@/lib/showToast";


const SHOWINGS_URL = "api/showings";

interface ConfirmDeleteShowingModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  showing: Showing;
  reservations: Reservation[]; // Wszystkie rezerwacje
}

export default function ConfirmDeleteShowingModal({
  isOpen,
  onClose,
  onConfirm,
  showing,
  reservations,
}: ConfirmDeleteShowingModalProps) {
  const auth = useAuth();
  const [hasReservations, setHasReservations] = useState(false);

  const checkReservations = () => {
    const reservationsForShowing = reservations.filter((res) => res.showing.id === showing.id);

    setHasReservations(reservationsForShowing.length > 0);
  };

  const handleDelete = async () => {
    if (hasReservations) {
      showToast("Nie można usunąć seansu z istniejącymi rezerwacjami!", true);

      return;
    }

    try {
      const response = await fetch(`${SHOWINGS_URL}/${showing.id}`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
      });

      if (response.status === 401) {
        auth.loginRequired();

        return;
      }

      if (!response.ok) {
        throw new Error("Nie udało się usunąć seansu!");
      }

      showToast("Seans usunięty pomyślnie!", false);
      onClose();
      onConfirm();
    } catch {
      showToast("Wystąpił błąd podczas usuwania seansu.", true);
    }
  };

  React.useEffect(() => {
    if (isOpen) {
      checkReservations();
    }
  }, [isOpen]);

  return (
    <Modal className="py-1" isOpen={isOpen} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <ModalContent>
        {() => (
          <>
            <ModalHeader>Potwierdź usunięcie seansu</ModalHeader>
            <ModalBody>
              {hasReservations ? (
                <p className="text-danger">
                  Nie można usunąć seansu <strong>{showing.movie.title}</strong>, ponieważ istnieją powiązane rezerwacje.
                </p>
              ) : (
                <p>
                  Czy na pewno chcesz usunąć seans <strong>{showing.movie.title}</strong> w sali{" "}
                  <strong>{showing.cinema_room.name}</strong>?
                </p>
              )}
            </ModalBody>
            <ModalFooter>
              <Button color="primary" variant="ghost" onPress={onClose}>
                Anuluj
              </Button>
              <Button
                color="danger"
                isDisabled={hasReservations}
                variant="ghost"
                onPress={handleDelete}
              >
                Usuń
              </Button>
            </ModalFooter>
          </>
        )}
      </ModalContent>
    </Modal>
  );
}

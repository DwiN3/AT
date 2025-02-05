"use client"

import React from "react";
import { Button } from "@nextui-org/button";
import { Modal, ModalBody, ModalContent, ModalFooter, ModalHeader } from "@heroui/modal";

import { useAuth } from "@/providers/authProvider";
import { showToast } from "@/lib/showToast";

const HALLS_URL = "api/halls";

interface ConfirmHallDeleteModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  hall: CinemaRoom;
}

export default function ConfirmHallDeleteModal({ isOpen, onClose, onConfirm, hall }: ConfirmHallDeleteModalProps) {
  const auth = useAuth();

  const handleDelete = async () => {

    try { 
      const response = await fetch(`${HALLS_URL}/${hall.id}`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
      });
  
      if (response.status === 401) {
        auth.loginRequired();
  
        return;
      }
  
      if (!response.ok) {
        throw new Error("Nie udało się usunąć sali kinowej!");
      }
  
      showToast("Sala kinowa usunięty pomyślnie!", false);
  
      onClose();
      onConfirm();
    } catch {
      showToast("Wystąpił błąd podczas usuwania sali kinowej.", true);
    }
  };

  return (
    <Modal className="py-1" isOpen={isOpen} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <ModalContent>
        {() => (
          <>
            <ModalHeader>Potwierdź usuwanie</ModalHeader>
            <ModalBody>
              <p>Czy na pewno chcesz usunąć salę kinową &quot;{hall.name}&quot;?</p>
            </ModalBody>
            <ModalFooter>
              <Button color="primary" variant="ghost" onPress={onClose}>
                Anuluj
              </Button>
              <Button color="danger" variant="ghost" onPress={() => handleDelete()}>
                Usuń
              </Button>
            </ModalFooter>
          </>
        )}
      </ModalContent>
    </Modal>
  );
}

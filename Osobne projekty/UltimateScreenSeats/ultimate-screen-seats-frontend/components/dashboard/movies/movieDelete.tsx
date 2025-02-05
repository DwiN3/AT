"use client"

import React from "react";
import { Button } from "@nextui-org/button";
import { Modal, ModalBody, ModalContent, ModalFooter, ModalHeader } from "@heroui/modal";

import { useAuth } from "@/providers/authProvider";
import { showToast } from "@/lib/showToast";

const MOVIES_URL = "api/movies";

interface ConfirmDeleteModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  movie: Movie;
}

export default function ConfirmDeleteModal({ isOpen, onClose, onConfirm, movie }: ConfirmDeleteModalProps) {
  const auth = useAuth();

  const handleDelete = async () => {

    try { 
      const response = await fetch(`${MOVIES_URL}/${movie.id}`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
      });
  
      if (response.status === 401) {
        auth.loginRequired();
  
        return;
      }
  
      if (!response.ok) {
        throw new Error("Nie udało się usunąć filmu!");
      }
  
      showToast("Film usunięty pomyślnie!", false);
  
      onClose();
      onConfirm();
    } catch {
      showToast("Wystąpił błąd podczas zapisywania filmu.", true);
    }
  };

  return (
    <Modal className="py-1" isOpen={isOpen} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <ModalContent>
        {() => (
          <>
            <ModalHeader>Potwierdź usuwanie</ModalHeader>
            <ModalBody>
              <p>Czy na pewno chcesz usunąć film &quot;{movie.title}&quot;?</p>
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

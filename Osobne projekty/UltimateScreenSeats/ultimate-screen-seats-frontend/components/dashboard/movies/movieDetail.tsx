"use client";

import React from "react";
import { Button } from "@nextui-org/button";
import { Modal, ModalBody, ModalContent, ModalFooter, ModalHeader } from "@heroui/modal";
import { Image } from "@nextui-org/image";


interface DetailsModalProps {
  isOpen: boolean;
  onClose: () => void;
  movie: Movie;
}

export default function DetailsModal({ isOpen, onClose, movie }: DetailsModalProps) {
  return (
    <Modal
      className="py-2"
      isOpen={isOpen}
      scrollBehavior='inside'
      size="xl"
      onOpenChange={(isOpen) => !isOpen && onClose()}
    >
      <ModalContent>
        {() => (
          <>
            <ModalHeader>
              <h1 className="text-2xl font-bold text-primary">{movie.title}</h1>
            </ModalHeader>
            <ModalBody>
              <div className="flex flex-col gap-4">
                <div className="flex gap-4">
                  {movie.image && (
                    <Image
                      alt={`${movie.title} Thumbnail`}
                      className="h-[12.5rem] object-cover rounded-lg"
                      src={movie.image}
                    />
                  )}
                  {movie.background_image && (
                    <Image
                      alt={`${movie.title} Background`}
                      className="h-[12.5rem] w-full object-cover rounded-lg"
                      src={movie.background_image}
                    />
                  )}
                </div>
                
                <div className="flex flex-col gap-1">
                  <p>
                    <strong>Opis:</strong> {movie.description}
                  </p>
                  <p>
                    <strong>Data wydania:</strong> {new Date(movie.release_date).toLocaleDateString()}
                  </p>
                  <p>
                    <strong>Gatunek:</strong> {movie.genre.map((g) => g.name).join(", ")}
                  </p>
                  <p>
                    <strong>Długość filmu (w minutach):</strong> {movie.movie_length} minutes
                  </p>
                  <p>
                    <strong>Sugerowany wiek:</strong> {movie.age_classification}+
                  </p>
                  <p>
                    <strong>Reżyser:</strong> {movie.director}
                  </p>
                  <p>
                    <strong>Obsada:</strong> {movie.cast}
                  </p>
                  {movie.trailer_url && (
                    <p>
                      <strong>Trailer:</strong>{" "}
                      <a
                        className="text-blue-500 underline hover:text-blue-700"
                        href={movie.trailer_url}
                        rel="noopener noreferrer"
                        target="_blank"
                      >
                        Zobacz trailer
                      </a>
                    </p>
                  )}
                </div>
              </div>
            </ModalBody>
            <ModalFooter>
              <Button color="primary" onPress={onClose}>
                Zamkij
              </Button>
            </ModalFooter>
          </>
        )}
      </ModalContent>
    </Modal>
  );
}

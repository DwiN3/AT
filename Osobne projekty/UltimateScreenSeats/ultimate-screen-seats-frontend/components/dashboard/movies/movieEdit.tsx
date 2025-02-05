"use client";

import React, { useState } from "react";
import { Modal, ModalBody, ModalContent, ModalFooter, ModalHeader, ModalProps } from "@heroui/modal";
import { Button } from "@nextui-org/button";
import { Input, Textarea } from "@nextui-org/input";
import { Image } from "@nextui-org/image";
import { Select, SelectItem } from "@nextui-org/select";

import { showToast } from "@/lib/showToast";
import { useAuth } from "@/providers/authProvider";

const MOVIES_URL = "api/movies";


interface EditModalProps {
  isOpen: boolean;
  onClose: () => void;
  movie?: Movie;
  onSave: (updatedMovie: Movie) => void;
  genres: Genre[];
}

export default function EditModal({ isOpen, onClose, movie, onSave, genres = [] }: EditModalProps) {
  const [formData, setFormData] = useState<Movie>(
    movie || {
      id: Date.now(),
      title: "",
      description: "",
      genre: [],
      movie_length: 0,
      age_classification: 0,
      image: "",
      release_date: "",
      trailer_url: "",
      cast: "",
      director: "",
      background_image: "",
    }
  );

  const [errors, setErrors] = useState<Record<string, boolean>>({});
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const auth = useAuth();

  const validateForm = (): boolean => {
    const newErrors: Record<string, boolean> = {};
    let hasErrors = false;

    if (!formData.title) {
      newErrors.title = true;
      hasErrors = true;
    }
    if (!formData.description) {
      newErrors.description = true;
      hasErrors = true;
    }
    if (!formData.release_date) {
      newErrors.release_date = true;
      hasErrors = true;
    }
    if (!formData.movie_length || formData.movie_length <= 0) {
      newErrors.movie_length = true;
      hasErrors = true;
    }
    if (!formData.age_classification || formData.age_classification <= 0) {
      newErrors.age_classification = true;
      hasErrors = true;
    }
    if (formData.genre.length === 0) {
      newErrors.genre = true;
      hasErrors = true;
    }
    if (!formData.image) {
      newErrors.image = true;
      hasErrors = true;
    }
    if (!formData.background_image) {
      newErrors.background_image = true;
      hasErrors = true;
    }
    if (!formData.cast) {
      newErrors.cast = true;
      hasErrors = true;
    }
    if (!formData.director) {
      newErrors.director = true;
      hasErrors = true;
    }
    if (!formData.trailer_url) {
      newErrors.trailer_url = true;
      hasErrors = true;
    }

    setErrors(newErrors);
    if (hasErrors) {
      setErrorMessage("Wypełnij wszystkie pola i spróbuj ponownie.");
    } else {
      setErrorMessage(null);
    }

    return !hasErrors;
  };

  const handleGenreChange = (selectedGenreIds: Set<string>) => {
    const genreIds = Array.from(selectedGenreIds).map((id) => parseInt(id));

    setFormData((prev) => ({
      ...prev,
      genre: genres.filter((g) => genreIds.includes(g.id)),
    }));
  };

  const handleSubmit = async () => {
    if (!validateForm()) {
      return;
    }

    try {
      const method = movie ? "PATCH" : "POST";
      const url = movie ? `${MOVIES_URL}/${movie.id}` : MOVIES_URL;

      const response = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...formData,
          genre_id: formData.genre.map((g) => g.id),
        }),
      });

      if (response.status === 401) {
        auth.loginRequired();

        return;
      }

      if (!response.ok) {
        throw new Error(movie ? "Nie udało się zaktualizować filmu!" : "Nie udało sie dodać filmu!");
      }

      showToast(movie ? "Film zaktualizowany pomyślnie!" : "Film utworzony pomyślnie!", false);

      onSave(formData);
      onClose();
    } catch {
      showToast("Wystąpił błąd podczas zapisywania filmu.", true);
    }
  };

  const handleChange = (key: keyof Movie, value: any) => {
    setFormData((prev) => ({ ...prev, [key]: value }));
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
            <ModalHeader className="text-2xl font-bold text-primary">{movie ? `Edytuj film ${movie.title}` : "Dodaj nowy film"}</ModalHeader>
            <ModalBody>
              {errorMessage && (
                <div className="text-danger font-bold text-lg text-center mb-4">{errorMessage}</div>
              )}
              <Input
                errorMessage={errors.title ? "To pole jest wymagane.": undefined}
                isInvalid={errors.title}
                label="Tytuł"
                value={formData.title}
                onChange={(e) => handleChange("title", e.target.value)}
              />
              <Textarea
                errorMessage={errors.description ? "To pole jest wymagane.": undefined}
                isInvalid={errors.description}
                label="Opis"
                value={formData.description}
                onChange={(e) => handleChange("description", e.target.value)}
              />
              <Input
                errorMessage={errors.release_date ? "To pole jest wymagane.": undefined}
                isInvalid={errors.release_date}
                label="Data wydania"
                type="date"
                value={formData.release_date}
                onChange={(e) => handleChange("release_date", e.target.value)}
              />
              <Input
                errorMessage={errors.movie_length ? "To pole jest wymagane.": undefined}
                isInvalid={errors.movie_length}
                label="Długość filmu (w minutach)"
                type="number"
                value={formData.movie_length.toString()}
                onChange={(e) =>
                  handleChange("movie_length", parseInt(e.target.value) || 0)
                }
              />
              <Input
                errorMessage={errors.age_classification ? "To pole jest wymagane.": undefined}
                isInvalid={errors.age_classification}
                label="Sugerowany wiek"
                type="number"
                value={formData.age_classification.toString()}
                onChange={(e) =>
                  handleChange("age_classification", parseInt(e.target.value) || 0)
                }
              />
              <Select
                errorMessage={errors.genre ? "Wybierz co najmniej jednen gatunek." : undefined}
                isInvalid={errors.genre}
                label="Gatunek/i"
                placeholder="Wybierz gatunek/i"
                selectedKeys={new Set(formData.genre.map((g) => g.id.toString()))}
                selectionMode="multiple"
                onSelectionChange={(selected) =>
                  handleGenreChange(selected as Set<string>)
                }
              >
                {genres.map((genre) => (
                  <SelectItem key={genre.id.toString()} value={genre.id.toString()}>
                    {genre.name}
                  </SelectItem>
                ))}
              </Select>
              <Input
                errorMessage={errors.director? "To pole jest wymagane.": undefined}
                isInvalid={errors.director}
                label="Reżyser"
                value={formData.director}
                onChange={(e) => handleChange("director", e.target.value)}
              />
              <Textarea
                errorMessage={errors.cast? "To pole jest wymagane.": undefined}
                isInvalid={errors.cast}
                label="Obsada"
                value={formData.cast}
                onChange={(e) => handleChange("cast", e.target.value)}
              />
              <Input
                errorMessage={errors.trailer_url? "To pole jest wymagane.": undefined}
                isInvalid={errors.trailer_url}
                label="Link to trailera"
                value={formData.trailer_url}
                onChange={(e) => handleChange("trailer_url", e.target.value)}
              />
              <div className="flex gap-3">
                {formData.image && (
                  <Image
                    alt="Movie Thumbnail"
                    className="w-[10rem] object-cover border rounded"
                    src={formData.image}
                  />
                )}
                <Textarea
                  errorMessage={errors.image ? "To pole jest wymagane.": undefined}
                  isInvalid={errors.image}
                  label="Zdjęcie okładki"
                  value={formData.image}
                  onChange={(e) => handleChange("image", e.target.value)}
                />
              </div>
              <div className="flex gap-3">
                {formData.background_image && (
                  <Image
                    alt="Background"
                    className="w-[24rem] object-cover border rounded"
                    src={formData.background_image}
                  />
                )}
                <Textarea
                  errorMessage={errors.background_image ? "To pole jest wymagane.": undefined}
                  isInvalid={errors.background_image}
                  label="Zdjęcie w tle"
                  value={formData.background_image}
                  onChange={(e) => handleChange("background_image", e.target.value)}
                />
              </div>
            </ModalBody>
            <ModalFooter className="mt-2">
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

"use client"

import { Button } from "@nextui-org/button";
import { useEffect, useState } from "react";
import { Select, SelectItem } from "@nextui-org/select";

import { showToast } from "@/lib/showToast";


const GENRES_URL = 'api/movies/genres'

interface FilterCarouselProps {
    activeFilter: string;
    onFilterChange: (filter: string) => void;
}

export default function FilterCarousel({ activeFilter, onFilterChange }: FilterCarouselProps) {
    const [genres, setGenres] = useState<Genre[]>([]);
    const [isSmallScreen, setIsSmallScreen] = useState(false);

    useEffect(() => {
        const fetchGenres = async () => {
            try {
                const response = await fetch(GENRES_URL, {
                    method: "GET",
                    headers: { "Content-Type": "application/json" },
                });

                if (!response.ok) {
                    throw new Error("Failed to fetch movies.");
                }

                const data = await response.json();

                const allGenres = [{ id: 0, name: "Wszystkie" }, ...data];

                setGenres(allGenres);
            } catch {
                showToast("Nie udało się pobrać gatunków filmów.", true);

                return null;
            }
        }

        fetchGenres();
    }, []);

    useEffect(() => {
        const handleResize = () => {
            setIsSmallScreen(window.innerWidth < 640);
        };

        handleResize();
        window.addEventListener("resize", handleResize);

        return () => window.removeEventListener("resize", handleResize);
    }, []);

    return (
        <div className="flex flex-col items-center gap-3 py-5 mt-5">
            {!isSmallScreen && (
                <div className="flex flex-row flex-wrap justify-center gap-3">
                    {genres.map(({ id, name }) => (
                        <Button
                            key={id}
                            className={`px-6 py-2 rounded ${activeFilter === name
                                    ? "bg-primary text-white"
                                    : "bg-default-300 text-default-700"
                                }`}
                            radius="lg"
                            onClick={() => onFilterChange(name)}
                        >
                            {name}
                        </Button>
                    ))}
                </div>
            )}

            {isSmallScreen && (
                <Select
                    className="w-full max-w-xs px-2"
                    label="Gatunek filmu"
                    labelPlacement="outside"
                    placeholder="Wybierz gatunek filmu"
                    selectedKeys={[activeFilter]}
                    variant="faded"
                    onChange={(e) => onFilterChange(e.target.value)}
                >
                    {genres.map(({ name }) => (
                        <SelectItem key={name} value={name}>
                            {name}
                        </SelectItem>
                    ))}
                </Select>
            )}
        </div>
    );
}
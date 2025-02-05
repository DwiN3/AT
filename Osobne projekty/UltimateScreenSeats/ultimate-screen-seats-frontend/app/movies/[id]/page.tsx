"use client"

import { Spinner } from "@nextui-org/spinner";
import { use, useEffect, useState } from "react";
import Image from "next/image";
import { CalendarDays, Film, LoaderPinwheel, UserRound, UsersRound } from "lucide-react";
import Link from "next/link";
import { Button } from "@nextui-org/button";
import { useRouter } from "next/navigation";
import { Modal, ModalBody, ModalContent, ModalFooter, ModalHeader, useDisclosure } from "@heroui/modal";

import MovieLayout from "./layout";

import { showToast } from "@/lib/showToast";
import { useAuth } from "@/providers/authProvider";

const MOVIES_URL = "/api/movies";


export default function MoviePage({ params }: { params: Promise<{ id: string }> }) {
    const { id } = use(params);
    const [movie, setMovie] = useState<Movie>();
    const [loading, setLoading] = useState(true);

    const { isOpen, onOpen, onOpenChange } = useDisclosure();

    const auth = useAuth();
    const router = useRouter();

    useEffect(() => {
        const fetchMovieDetails = async () => {
            setLoading(true);
            try {
                const response = await fetch(`${MOVIES_URL}/${id}`, {
                    method: "GET",
                    headers: { "Content-Type": "application/json" },
                });

                if (response.status === 401) {
                    auth.loginRequired();

                    return;
                }

                if (response.status === 404) {
                    router.push('/not-found');

                    return;
                }

                if (!response.ok) {
                    throw new Error("Failed to fetch movie details.");
                }

                const data = await response.json();

                setMovie(data);
            } catch {
                showToast("Nie udało się pobrać detali filmu.", true);
            } finally {
                setLoading(false);
            }
        };

        fetchMovieDetails();
    }, [id, auth]);


    const handleShowShowings = () => {
        if (!auth.isAuthenticated) {
            onOpen();

            return;
        }

        router.push(`/movies/${id}/showings`);
    }

    const handleLogin = () => {
        router.push('/login')
    }

    if (loading) {
        return (
            <div className="flex flex-row gap-4 h-full justify-center items-center">
                <Spinner />
                <p className="text-md">Ładowanie detali filmu...</p>
            </div>
        );
    }

    if (!movie) {
        return <p>Nie znaleziono filmu.</p>;
    }

    return (
        <MovieLayout backgroundImage={movie.background_image}>
            <div className="max-w-7xl mx-auto p-8 lg:h-[85svh] flex lg:flex-row flex-col items-end lg:gap-10 gap-4">
                <div className="lg:w-2/5 w-full relative lg:h-full sm:h-[36svh] h-[20svh]">
                    <Image
                        fill
                        priority
                        alt="Movie image"
                        className="md:pt-10 object-contain object-bottom"
                        src={movie.image}
                    />
                </div>
                <div className="flex-1 text-white flex flex-col gap-1 justify-start overflow-auto">
                    <h1 className="sm:text-4xl text-2xl text-primary font-bold mb-4">{movie.title}</h1>
                    <div className="overflow-auto lg:max-h-full sm:max-h-[30svh] max-h-[48svh]">
                        <p className="sm:text-base text-sm sm:mb-8 mb-6 sm:text-start text-justify">{movie.description}{movie.description}</p>

                        <p className="sm:text-md text-xs flex gap-4 items-center">
                            <UserRound className="w-[20px] flex-shrink-0" />
                            <span className="font-bold">Reżyser:</span> {movie.director}
                        </p>

                        <p className="sm:text-md text-xs flex gap-4 items-center">
                            <LoaderPinwheel className="w-[20px] flex-shrink-0" />
                            <span className="font-bold">Gatunek:</span> {movie.genre.map((g) => g.name).join(', ')}
                        </p>

                        <p className="sm:text-md text-xs flex gap-4 items-center">
                            <CalendarDays className="w-[20px] flex-shrink-0" />
                            <span className="font-bold">Data wydania:</span> {movie.release_date}
                        </p>

                        <p className="sm:text-md text-xs flex gap-4 items-center">
                            <UsersRound className="w-[20px] flex-shrink-0" />
                            <span className="font-bold">Obsada:</span> {movie.cast}
                        </p>

                        <p className="sm:text-md text-xs flex gap-4 items-center">
                            <Film className="w-[20px] flex-shrink-0" />
                            <span className="font-bold">Trailer:</span>
                            <Link className="transition-colors-opacity hover:text-primary" href={movie.trailer_url}>
                                {movie.trailer_url}
                            </Link>
                        </p>
                    </div>

                    <Button className="sm:mt-8 mt-6" color="primary" size="lg" variant="ghost" onClick={handleShowShowings}>
                        Sprawdź seanse
                    </Button>
                </div>
            </div>

            <Modal backdrop="blur" isOpen={isOpen} size="lg" onOpenChange={onOpenChange}>
                <ModalContent>
                    {(onClose) => (
                        <>
                            <ModalHeader className="flex flex-col gap-1 text-2xl text-danger">Autoryzacja wymagana</ModalHeader>
                            <ModalBody>
                                <p>
                                    Aby przejść do strony rezerwacji miejsc musisz być zalogowany.
                                </p>
                            </ModalBody>
                            <ModalFooter>
                                <Button color="danger" variant="light" onPress={onClose}>
                                    Anuluj
                                </Button>
                                <Button color="primary" onPress={handleLogin}>
                                    Zaloguj się
                                </Button>
                            </ModalFooter>
                        </>
                    )}
                </ModalContent>
            </Modal>
        </MovieLayout>
    )
}

"use client";

import { useEffect, useState } from "react";
import { Card, CardBody, CardHeader } from "@nextui-org/card";
import { Button } from "@nextui-org/button";

import { Reservation } from "@/app/interfaces/reservation";
import { useAuth } from "@/providers/authProvider";
import ApiProxy from "@/app/api/proxy";

const RESERVATIONS_URL = "/api/reservation";


export default function Reservations() {
    const [reservations, setReservations] = useState<Reservation[]>([]);
    const [error, setError] = useState<string | null>(null);
    const { authToken, userId } = useAuth();

    const fetchReservations = async () => {
        if (!authToken || !userId) {
            setError("Nieprawidłowy token lub brak userId.");

            return;
        }
        const url = `${RESERVATIONS_URL}?user_id=${userId}`;

        try {
            const { data, status, error } = await ApiProxy.get(url, true);

            if (status === 200) {
                setReservations(Array.isArray(data) ? data : []);
            } else {
                setError(error?.message || "Wystąpił błąd przy pobieraniu rezerwacji.");
            }
        } catch {
            setError("Wystąpił błąd podczas pobierania danych.");
        }
    };

    useEffect(() => {
        fetchReservations();
    }, [authToken, userId]);

    const handleDelete = async (reservationId: number) => {
        const url = `${RESERVATIONS_URL}/${reservationId}`;

        try {
            const { status, error } = await ApiProxy.delete(url, true);

            if (status === 200) {
                setReservations((prevReservations) =>
                    prevReservations.filter((reservation) => reservation.id !== reservationId)
                );
            } else {
                setError(error?.message || "Nie udało się usunąć rezerwacji. Spróbuj ponownie.");
            }
        } catch {
            setError("Błąd podczas usuwania rezerwacji.");
        }
    };

    if (error) {
        return (
            <div className="flex justify-center items-center h-screen max-w-7xl mx-auto">
                <Card className="w-full max-w-lg p-8 bg-red-500 text-white rounded-lg shadow-lg">
                    <CardHeader className="p-2 flex-col items-start">
                        <h1 className="text-2xl font-semibold mb-2">Wystąpił błąd</h1>
                    </CardHeader>
                    <CardBody>
                        <p className="text-lg mb-4">{error}</p>
                        <Button
                            className="w-full mt-4"
                            color="default"
                            onClick={() => window.location.reload()}
                        >
                            Odśwież stronę
                        </Button>
                    </CardBody>
                </Card>
            </div>
        );
    }

    return (
        <div className="flex flex-col justify-center mt-10 max-w-7xl mx-auto sm:px-8 px-4">
            <Card className="w-full p-8">
                <CardHeader className="p-2 flex-col items-start border-b-2 border-default-200 mb-6">
                    <h1 className="text-primary text-4xl font-semibold mb-2">Moje Rezerwacje</h1>
                    <h2 className="text-default-500 text-lg">Wszystkie twoje rezerwacje w jednym miejscu.</h2>
                </CardHeader>
                <CardBody className="overflow-hidden flex flex-col p-0">
                    {reservations.length > 0 ? (
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                            {reservations.map((reservation) => (
                                <Card key={reservation.id} className="bg-gray-900 text-white p-6 rounded-lg shadow-lg hover:shadow-xl transition-shadow duration-300">
                                    <CardHeader>
                                        <h2 className="text-xl font-semibold mb-2">{reservation.showing.movie.title}</h2>
                                    </CardHeader>
                                    <CardBody>
                                        <p className="mb-2">
                                            <strong>Data:</strong> {new Date(reservation.showing.date).toLocaleString()}
                                        </p>
                                        <p className="mb-2">
                                            <strong>Rząd:</strong> {reservation.seat_row}, <strong>Miejsce:</strong> {reservation.seat_column}
                                        </p>
                                        <p className="mb-4">
                                            <strong>Cena:</strong> {reservation.showing.ticket_price} PLN
                                        </p>
                                        <Button
                                            className="w-full"
                                            color="danger"
                                            size="sm"
                                            onClick={() => handleDelete(reservation.id)}
                                        >
                                            Zrezygnuj z rezerwacji
                                        </Button>
                                    </CardBody>
                                </Card>
                            ))}
                        </div>
                    ) : (
                        <p className="text-center text-gray-400">Brak rezerwacji.</p>
                    )}
                </CardBody>
            </Card>
        </div>
    );
}
"use client"

import { useEffect, useState } from "react"
import { Card, CardBody, CardFooter, CardHeader } from "@nextui-org/card"
import { Button } from "@nextui-org/button"

import ApiProxy from "@/app/api/proxy";
import './Showing.tsx.css';

const SHOWING_URL = "/api/showing";
const RESERVATIONS_URL = "/api/reservation";

type SeatLayout = number[][];

export default function Showing({ id }: { id: number }) {
  const [seatLayout, setSeatLayout] = useState<SeatLayout | null>(null)
  const [selectedSeats, setSelectedSeats] = useState<{ row: number; column: number }[]>([])
  const [error, setError] = useState<string | null>(null)

  const fetchAvailableSeats = async () => {
    try {
      const res = await fetch(`${SHOWING_URL}/${id}`);
      const data = await res.json();

      if (res.ok) {
        const reservation = data;

        if (reservation?.cinema_room?.seat_layout) {
          setSeatLayout(reservation.cinema_room.seat_layout);
        } else {
          setError("Brak układu sali w odpowiedzi serwera.");
        }
      } else {
        setError(data?.message || "Błąd podczas pobierania dostępnych miejsc.");
      }
    } catch {
      setError("Wystąpił błąd podczas pobierania danych.");
    }
  };
  
  const toggleSeatSelection = (row: number, column: number) => {
    const seatIndex = selectedSeats.findIndex(
      (seat) => seat.row === row && seat.column === column
    );

    if (seatIndex >= 0) {
      setSelectedSeats((prev) => prev.filter((_, index) => index !== seatIndex));
    } else {
      setSelectedSeats((prev) => [...prev, { row, column }]);
    }
  };
  
  const reserveSeats = async () => {
    if (selectedSeats.length === 0) {
      setError("Nie wybrano żadnych miejsc.");

      return;
    }

    try {
      for (const seat of selectedSeats) {
        const { status, error } = await ApiProxy.post(RESERVATIONS_URL, {
          showing_id: id,
          seat_row: seat.row + 1,
          seat_column: seat.column + 1,
        }, true);

        if (status !== 201) {
          setError(error?.message || "Błąd przy rezerwowaniu miejsca.");

          return;
        }
      }

      setSelectedSeats([]);
      setError(null);
    } catch {
      setError("Wystąpił błąd podczas rezerwacji.");
    }
  };

  useEffect(() => {
    fetchAvailableSeats();
  }, [id]);

  if (error) {
    return (
      <div className="flex justify-center items-center h-screen">
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
    )
  }

  return (
    <div className="flex flex-col items-center mt-10">
      <Card className="w-full p-8 max-w-4xl">
        <CardHeader className="p-2 flex-col items-start border-b-2 border-default-200 mb-4">
          <h1 className="text-primary text-4xl font-semibold mb-2">Układ Sali Kinowej</h1>
          <h2 className="text-default-500 text-lg">Wybierz swoje miejsca.</h2>
        </CardHeader>
        <CardBody className="overflow-hidden flex flex-col">
          {seatLayout ? (
            <div className="flex flex-col items-center">
              <div className="w-full h-3 bg-black mb-8 flex justify-center items-center text-white font-bold text-lg rounded-lg" />
              <div className="grid gap-8">
                {seatLayout.map((row, rowIndex) => (
                  <div key={rowIndex} className="flex justify-center space-x-8">
                    {row.map((seat, columnIndex) => {
                      if (seat === -1) {
                        return (
                          <div
                            key={`${rowIndex}-${columnIndex}`}
                            className="w-16 h-16 bg-transparent"
                            style={{ visibility: "hidden" }}
                           />
                        );
                      }

                      const isSelected = selectedSeats.some(
                        (selectedSeat) =>
                          selectedSeat.row === rowIndex && selectedSeat.column === columnIndex
                      )

                      const isOccupied = seat === 1
                      const isFree = seat === 0

                      let seatClass = "cursor-pointer w-16 h-16"

                      if (isOccupied) {
                        seatClass = "bg-gray-500 cursor-not-allowed shadow-xl"
                      } else if (isSelected) {
                        seatClass = "bg-blue-500"
                      } else if (isFree) {
                        seatClass = "bg-green-500"
                      }

                      return (
                        <div key={`${rowIndex}-${columnIndex}`} className="flex items-center justify-center">
                          <input
                            checked={isSelected}
                            className={`checkbox w-10 h-10 cursor-pointer appearance-none border-2 border-gray-500 rounded-lg ${seatClass}`}
                            disabled={isOccupied || seat === -1}
                            type="checkbox"
                            onChange={() => toggleSeatSelection(rowIndex, columnIndex)}
                          />
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-center text-gray-400">Ładowanie układu sali...</p>
          )}
        </CardBody>
        <CardFooter className="flex justify-center mt-2">
          <Button className="mt-6 w-full max-w-lg" onClick={reserveSeats}>
            Zarezerwuj wybrane miejsca
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}

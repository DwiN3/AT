"use server";

import { NextResponse } from "next/server";
import ApiProxy from "../proxy";

const DJANGO_API_RESERVATIONS_URL = "http://127.0.0.1:8000/api/reservation";

export async function GET(request: Request) {
    const url = new URL(request.url);
    const userId = url.searchParams.get("user_id");

    if (!userId) {
        return NextResponse.json({ error: 'Brak ID użytkownika w zapytaniu.' }, { status: 400 });
    }

    try {
        const { data, status } = await ApiProxy.get(
            `${DJANGO_API_RESERVATIONS_URL}?user_id=${userId}`,
            true
        );

        if (!Array.isArray(data) || data.length === 0) {
            return NextResponse.json([], { status: 200 });
        }

        return NextResponse.json(data, { status });
    } catch (error) {
        console.error("Błąd przy pobieraniu rezerwacji:", error);
        return NextResponse.json(
            { error: "Błąd przy pobieraniu rezerwacji." },
            { status: 500 }
        );
    }
}

export async function POST(request: Request) {
    try {
      const body = await request.json();
  
      const { showing_id, seat_row, seat_column } = body;
      if (showing_id === undefined || seat_row === undefined || seat_column === undefined) {
        return NextResponse.json(
          { error: "Brak wymaganych danych w zapytaniu (showing_id, seat_row, seat_column)." },
          { status: 400 }
        );
      }
      const reservationData = {
        showing_id,
        seat_row,
        seat_column,
      };
  
      const { data, status } = await ApiProxy.post(DJANGO_API_RESERVATIONS_URL, reservationData, true);
  
      if (status === 201) {
        return NextResponse.json(data, { status: 201 });
      }
  
      return NextResponse.json({ error: "Błąd przy tworzeniu rezerwacji." }, { status: 500 });
    } catch (error) {
      console.error("Błąd przy tworzeniu rezerwacji:", error);
      return NextResponse.json({ error: "Błąd przy przetwarzaniu rezerwacji." }, { status: 500 });
    }
  }

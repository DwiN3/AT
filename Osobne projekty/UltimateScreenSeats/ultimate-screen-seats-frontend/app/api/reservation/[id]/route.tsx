"use server";

import { NextResponse } from "next/server";
import ApiProxy from "../../proxy";

const DJANGO_API_RESERVATIONS_URL = "http://127.0.0.1:8000/api/reservation";

export async function DELETE(request: Request, { params }: { params: { id: string } }) {
    const { id: reservationId } = params;

    if (!reservationId) {
        return NextResponse.json({ error: "Brak ID rezerwacji w zapytaniu." }, { status: 400 });
    }

    try {
        const { status } = await ApiProxy.delete(
            `${DJANGO_API_RESERVATIONS_URL}/${reservationId}`,
            true
        );

        if (status === 200) {
            return NextResponse.json(
                { message: "Rezerwacja została usunięta." },
                { status }
            );
        } else {
            return NextResponse.json(
                { error: "Nie udało się usunąć rezerwacji." },
                { status }
            );
        }
    } catch (error) {
        console.error("Błąd przy usuwaniu rezerwacji:", error);
        return NextResponse.json(
            { error: "Błąd przy usuwaniu rezerwacji." },
            { status: 500 }
        );
    }
}

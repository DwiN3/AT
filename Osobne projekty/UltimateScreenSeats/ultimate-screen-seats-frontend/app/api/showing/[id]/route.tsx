"use server";

import { NextResponse } from "next/server";
import ApiProxy from "../../proxy";

const DJANGO_API_SHOWING_URL = "http://127.0.0.1:8000/api/showing/";

export async function GET(request: Request, context: { params: { id: string } }) {
    const { id } = context.params;

    if (!id) {
        return NextResponse.json(
            { error: "Brak ID pokazu w zapytaniu." },
            { status: 400 }
        );
    }

    try {
        const { data, status } = await ApiProxy.get(`${DJANGO_API_SHOWING_URL}${id}`, true);

        if (!data) {
            return NextResponse.json(
                { error: "Nie znaleziono danych dla podanego ID pokazu." },
                { status: 404 }
            );
        }

        return NextResponse.json(data, { status });
    } catch (error) {
        console.error("Błąd przy pobieraniu danych pokazu:", error);
        return NextResponse.json(
            { error: "Wystąpił błąd podczas pobierania danych pokazu." },
            { status: 500 }
        );
    }
}

"use server";

import { NextResponse } from "next/server";

import ApiProxy from "../../proxy";

const DJANGO_API_SHOWINGS_URL = "http://127.0.0.1:8000/api/reservation";


export async function GET(request: Request, context: { params: { option: string } }) {
    const { params } = context;
    const { option } = await params;
    const { searchParams } = new URL(request.url);

    const limit = searchParams.get("limit");

    let url;

    if (option) {
        url = new URL(`${DJANGO_API_SHOWINGS_URL}/${option}`);
    } else {
        url = new URL(DJANGO_API_SHOWINGS_URL);
    }

    if (limit) url.searchParams.append("limit", limit);

    const { data, status } = await ApiProxy.get(url.toString(), true);

    return NextResponse.json(data, { status });
}
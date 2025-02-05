"use server";

import { NextResponse } from "next/server";

import ApiProxy from "../proxy";

const DJANGO_API_HALLS_URL = "http://127.0.0.1:8000/api/cinema-room";

export async function GET(request: Request) {
    const { searchParams } = new URL(request.url);

    const limit = searchParams.get("limit");

    const url = new URL(DJANGO_API_HALLS_URL);

    if (limit) url.searchParams.append("limit", limit);

    const { data, status } = await ApiProxy.get(url.toString(), true);

    return NextResponse.json(data, { status });
}


export async function POST( request: Request) {
    const requestData = await request.json();

    const { data, status } = await ApiProxy.post(`${DJANGO_API_HALLS_URL}`, requestData, true);

    return NextResponse.json(data, { status: status });
}
"use server";

import { NextResponse } from "next/server";

import ApiProxy from "../../proxy";

const DJANGO_API_SHOWINGS_URL = "http://127.0.0.1:8000/api/showing/list";

export async function GET(request: Request) {
    const { searchParams } = new URL(request.url);

    const startDate = searchParams.get("start_date");
    const endDate = searchParams.get("end_date");

    const url = new URL(DJANGO_API_SHOWINGS_URL);

    if (startDate) url.searchParams.append("start_date", startDate);
    if (endDate) url.searchParams.append("end_date", endDate);

    const { data, status } = await ApiProxy.get(url.toString(), false);

    return NextResponse.json(data, { status });
}

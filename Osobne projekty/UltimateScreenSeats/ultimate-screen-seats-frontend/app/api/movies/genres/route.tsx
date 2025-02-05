"use server"

import { NextResponse } from "next/server";

import ApiProxy from "../../proxy";

const DJANGO_API_MOVIES_URL = "http://127.0.0.1:8000/api/movie/genre"


export async function GET() {
    const {data, status} = await ApiProxy.get(DJANGO_API_MOVIES_URL, false);

    return NextResponse.json(data, {status: status});
}
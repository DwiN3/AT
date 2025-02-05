"use server"

import { NextResponse } from "next/server";

import ApiProxy from "../../proxy";

const DJANGO_API_REGISTER_URL = "http://127.0.0.1:8000/api/auth/register"

export async function POST(request: Request) {
    const requestData = await request.json();
    const { data, status } = await ApiProxy.post(DJANGO_API_REGISTER_URL, requestData, false);

    return NextResponse.json(data, { status: status });
}
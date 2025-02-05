"use server"

import { NextResponse } from "next/server";

import ApiProxy from "../../proxy";

const DJANGO_API_SHOWINGS_URL = "http://127.0.0.1:8000/api/showing/"


export async function DELETE( request: Request, context: { params: { id: string } }) {
    const { params } = context;
    const { id } = await params;

    const { data, status } = await ApiProxy.delete(`${DJANGO_API_SHOWINGS_URL}${id}`, true);

    return NextResponse.json(data, { status: status });
}
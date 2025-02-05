"use server"

import { NextResponse } from "next/server";

import ApiProxy from "../../proxy";

const DJANGO_API_MOVIES_URL = "http://127.0.0.1:8000/api/movie/"


export async function GET( request: Request, context: { params: { id: string } }) {
    const { params } = context;
    const { id } = await params;

    const { data, status } = await ApiProxy.get(`${DJANGO_API_MOVIES_URL}${id}`, false);

    return NextResponse.json(data, { status: status });
}

export async function PATCH( request: Request, context: { params: { id: string } }) {
    const { params } = context;
    const { id } = await params;
    const requestData = await request.json();

    const { data, status } = await ApiProxy.patch(`${DJANGO_API_MOVIES_URL}${id}`, requestData, true);

    return NextResponse.json(data, { status: status });
}

export async function DELETE( request: Request, context: { params: { id: string } }) {
    const { params } = context;
    const { id } = await params;

    const { data, status } = await ApiProxy.delete(`${DJANGO_API_MOVIES_URL}${id}`, true);

    return NextResponse.json(data, { status: status });
}
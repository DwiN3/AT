"use server"

import { NextResponse } from "next/server"

import ApiProxy from "../../proxy";

import { setRefreshToken, setRole, setToken } from "@/lib/authServer";


const DJANGO_API_LOGIN_URL = "http://127.0.0.1:8000/api/auth/login"

interface LoginResponse {
    access: string;
    refresh: string;
    role: string;
}

export async function POST(request: Request) {
    const requestData = await request.json();
    const { data, status } = await ApiProxy.post(DJANGO_API_LOGIN_URL, requestData, false);

    if (status === 200) {
        const loginData = data as LoginResponse;

        setToken(loginData.access);
        setRefreshToken(loginData.refresh);
        setRole(loginData.role);
    }

    return NextResponse.json(data, { status: status });
}
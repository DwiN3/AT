"use server"

import { cookies } from "next/headers";

const TOKEN_AGE = 3600 * 24;
const TOKEN_NAME = "auth-token";
const TOKEN_REFRESH_NAME = "auth-refresh-token";
const ROLE_NAME = "role";


export async function getToken() {
    const cookieStore = await cookies();

    return cookieStore.get(TOKEN_NAME)?.value || null;
}

export async function getRefreshToken() {
    const cookieStore = await cookies();

    return cookieStore.get(TOKEN_REFRESH_NAME)?.value || null;
}

export async function getRole() {
    const cookieStore = await cookies();

    return cookieStore.get(ROLE_NAME)?.value || null;
}

export async function setToken(authToken: string | null | undefined) {
    const cookieStore = await cookies();

    return cookieStore.set({
        name: TOKEN_NAME,
        value: authToken ?? "",
        httpOnly: true,
        sameSite: 'strict',
        secure: process.env.NODE_ENV !== 'development',
        maxAge: TOKEN_AGE,
    });
}

export async function setRefreshToken(authRefreshToken: string | null | undefined) {
    const cookieStore = await cookies();

    return cookieStore.set({
        name: TOKEN_REFRESH_NAME,
        value: authRefreshToken ?? "",
        httpOnly: true,
        sameSite: 'strict',
        secure: process.env.NODE_ENV !== 'development',
        maxAge: TOKEN_AGE * 7,
    });
}

export async function setRole(role: string | null | undefined) {
    const cookieStore = await cookies();

    return cookieStore.set({
        name: ROLE_NAME,
        value: role ?? "",
        httpOnly: true,
        sameSite: 'strict',
        secure: process.env.NODE_ENV !== 'development',
        maxAge: TOKEN_AGE,
    });
}

export async function deleteTokens() {
    const cookieStore = await cookies();

    await cookieStore.delete(TOKEN_NAME);
    await cookieStore.delete(TOKEN_REFRESH_NAME);
    await cookieStore.delete(ROLE_NAME);
}
"use client"

import { jwtDecode } from "jwt-decode";

const TOKEN_NAME = "auth-token";
const TOKEN_REFRESH_NAME = "auth-refresh-token";


export function getToken(): string | null {
    return localStorage.getItem(TOKEN_NAME);
}

export function setToken(authToken: string): void {
    localStorage.setItem(TOKEN_NAME, authToken);
}

export function getRefreshToken(): string | null {
    return localStorage.getItem(TOKEN_REFRESH_NAME);
}

export function setRefreshToken(authRefreshToken: string): void {
    localStorage.setItem(TOKEN_REFRESH_NAME, authRefreshToken);
}

export function deleteTokens(): void {
    localStorage.removeItem(TOKEN_NAME);
    localStorage.removeItem(TOKEN_REFRESH_NAME);
}

export function isTokenExpired(token: string): boolean {
    try {
        const decoded: { exp?: number } = jwtDecode(token);

        return decoded.exp ? decoded.exp < Date.now() / 1000 : false;
    } catch {
        return true;
    }
}
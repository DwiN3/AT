"use client"

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState, createContext, useContext } from "react"

import { setToken, setRefreshToken, deleteTokens, isTokenExpired } from "@/lib/authClient";
import { deleteTokens as deleteServerTokens } from "@/lib/authServer";

interface AuthContextProps {
    isAuthenticated: boolean
    authToken: string | null
    username: string
    role: string
    userId: number | null
    login: (username?: string, role?: string, authToken?: string, refreshToken?: string) => void
    logout: () => void
    loginRequired: () => void
}

const AuthContext = createContext<AuthContextProps | null>(null)

const LOGIN_REDIRECT_URL = "/"
const LOGOUT_REDIRECT_URL = "/"
const LOGIN_REQUIRED_URL = "/login"

const LOCAL_TOKEN_KEY = "auth-token"
const LOCAL_USERNAME_KEY = "username"
const LOCAL_ROLE_KEY = "role"
const LOCAL_USER_ID_KEY = "user-id"

interface AuthProviderProps {
    children: React.ReactNode
}

export function AuthProvider({ children }: AuthProviderProps) {
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [username, setUsername] = useState("");
    const [role, setRole] = useState("");
    const [userId, setUserId] = useState<number | null>(null);
    const [authToken, setAuthToken] = useState<string | null>(null);

    const router = useRouter();
    const pathname = usePathname();
    const searchParams = useSearchParams();

    useEffect(() => {
        const checkToken = async () => {
            const token = localStorage.getItem(LOCAL_TOKEN_KEY);

            if (token) {
                if (isTokenExpired(token)) {
                    loginRequired();

                    return;
                }

                setIsAuthenticated(true);
                setAuthToken(token);

                const decodedToken = decodeToken(token);
                const userIdFromToken = decodedToken?.user_id;
                
                if (userIdFromToken) {
                    setUserId(userIdFromToken);
                    localStorage.setItem(LOCAL_USER_ID_KEY, String(userIdFromToken));
                }
            } else {
                // loginRequired();

                return;
            }

            const storedUsername = localStorage.getItem(LOCAL_USERNAME_KEY);

            if (storedUsername) {
                setUsername(storedUsername);
            }

            const storedRole = localStorage.getItem(LOCAL_ROLE_KEY);

            if (storedRole) {
                setRole(storedRole);
            }
        }

        checkToken();
    }, []);

    const login = async (username?: string, role?: string, authToken?: string, refreshToken?: string) => {
        if (authToken) {
            setToken(authToken);
            setAuthToken(authToken);
            setIsAuthenticated(true);

            const decodedToken = decodeToken(authToken);
            const userIdFromToken = decodedToken?.user_id;
            if (userIdFromToken) {
                setUserId(userIdFromToken);
                localStorage.setItem(LOCAL_USER_ID_KEY, String(userIdFromToken));
            }
        }

        if (refreshToken) {
            setRefreshToken(refreshToken);
        }

        if (username) {
            localStorage.setItem(LOCAL_USERNAME_KEY, username);
            setUsername(username);
        } else {
            localStorage.removeItem(LOCAL_USERNAME_KEY);
            setUsername("");
        }

        if (role) {
            localStorage.setItem(LOCAL_ROLE_KEY, role);
            setRole(role);
        } else {
            localStorage.removeItem(LOCAL_ROLE_KEY);
            setRole("");
        }

        const nextUrl = searchParams.get("next");
        const invalidNextUrls = ["/login", "/logout", "/register"];
        const nextValidUrl = nextUrl && nextUrl.startsWith("/") && !invalidNextUrls.includes(nextUrl);

        if (nextValidUrl) {
            router.replace(nextUrl);
        } else {
            router.replace(LOGIN_REDIRECT_URL);
        }
    };

    const logout = () => {
        setIsAuthenticated(false);
        deleteTokens();
        deleteServerTokens();
        localStorage.removeItem(LOCAL_USERNAME_KEY);
        localStorage.removeItem(LOCAL_ROLE_KEY);
        localStorage.removeItem(LOCAL_USER_ID_KEY);

        router.replace(LOGOUT_REDIRECT_URL);
    }

    const loginRequired = () => {
        setIsAuthenticated(false);
        deleteTokens();
        deleteServerTokens();
        localStorage.removeItem(LOCAL_USERNAME_KEY);
        localStorage.removeItem(LOCAL_ROLE_KEY);
        localStorage.removeItem(LOCAL_USER_ID_KEY);
        const loginWithNextUrl = `${LOGIN_REQUIRED_URL}?next=${pathname}`;

        router.replace(loginWithNextUrl);
    };

    const decodeToken = (token: string) => {
        try {
            const payload = token.split(".")[1];
            const decoded = JSON.parse(atob(payload));
            return decoded;
        } catch (e) {
            return null;
        }
    };

    return (
        <AuthContext.Provider value={{ isAuthenticated, authToken, login, logout, loginRequired, username, role, userId }}>
            {children}
        </AuthContext.Provider>
    )
}

export function useAuth() {
    const context = useContext(AuthContext)

    if (!context) {
        throw new Error("useAuth must be used within an AuthProvider")
    }

    return context
}
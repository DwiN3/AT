import { getToken } from "@/lib/authServer";


interface ApiResponse<T> {
    data: T | null;
    status: number;
    error?: { message: string; error?: unknown };
}

type CustomHeaders = Record<string, string>;

export default class ApiProxy {
    static async getHeaders(requireAuth: boolean): Promise<HeadersInit> {
        const headers: CustomHeaders = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        };

        const authToken = await getToken();

        if (authToken && requireAuth) {
            headers["Authorization"] = `Bearer ${authToken}`
        }

        return headers;
    }

    static async handleFetch<T>(endpoint: string, requestOptions: RequestInit): Promise<ApiResponse<T>> {
        let data: T | null = null;
        let status = 500;
        let error: { message: string; error?: unknown } | undefined;

        try {
            const response = await fetch(endpoint, requestOptions);

            data = await response.json();
            status = response.status;
        } catch (err) {
            error = { message: "Cannot reach API server", error: err };
            status = 500;
        }

        return { data, status, error };
    }

    static async post<T>(endpoint: string, object: Record<string, any>, requireAuth: boolean): Promise<ApiResponse<T>> {
        const jsonData = JSON.stringify(object);
        const headers = await ApiProxy.getHeaders(requireAuth);

        const requestOptions: RequestInit = {
            method: "POST",
            headers,
            body: jsonData,
        };

        return ApiProxy.handleFetch<T>(endpoint, requestOptions);
    }

    static async get(endpoint: string, requireAuth: boolean) {
        const headers = await ApiProxy.getHeaders(requireAuth);

        const requestOptions = {
            method: "GET",
            headers: headers
        };

        return await ApiProxy.handleFetch(endpoint, requestOptions);
    }

    static async patch<T>(endpoint: string, object: Record<string, any>, requireAuth: boolean): Promise<ApiResponse<T>> {
        const jsonData = JSON.stringify(object);
        const headers = await ApiProxy.getHeaders(requireAuth);

        const requestOptions: RequestInit = {
            method: "PATCH",
            headers,
            body: jsonData,
        };

        return ApiProxy.handleFetch<T>(endpoint, requestOptions);
    }

    static async put<T>(endpoint: string, object: Record<string, any>, requireAuth: boolean): Promise<ApiResponse<T>> {
        const jsonData = JSON.stringify(object);
        const headers = await ApiProxy.getHeaders(requireAuth);

        const requestOptions: RequestInit = {
            method: "PUT",
            headers,
            body: jsonData,
        };

        return ApiProxy.handleFetch<T>(endpoint, requestOptions);
    }

    static async delete<T>(endpoint: string, requireAuth: boolean): Promise<ApiResponse<T>> {
        const headers = await ApiProxy.getHeaders(requireAuth);
    
        const requestOptions: RequestInit = {
            method: "DELETE",
            headers,
        };
    
        return ApiProxy.handleFetch<T>(endpoint, requestOptions);
    }    
}

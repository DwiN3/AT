"use server";

import { jwtDecode } from "jwt-decode";
import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";

const protectedRoutes = [
  { path: /^\/movies\/\d+\/showings$/, roles: ['USER', 'ADMIN'] },
  { path: "/profile", roles: ['USER', 'ADMIN'] },
  { path: "/reservations", roles: ['USER', 'ADMIN'] },
  { path: "/dashboard", roles: ['ADMIN'] },
];

function isTokenExpired(token: string): boolean {
  try {
    const decoded: { exp?: number } = jwtDecode(token);

    return decoded.exp ? decoded.exp < Date.now() / 1000 : true;
  } catch {
    return true;
  }
}

export default async function middleware(request: NextRequest) {
  const path = request.nextUrl.pathname;
  const cookieStore = await cookies();

  const userRole = cookieStore.get("role")?.value;
  const token = cookieStore.get("auth-token")?.value;
  const isAuthenticated = token && !isTokenExpired(token);

  const matchedRoute = protectedRoutes.find((route) => {
    if (typeof route.path === "string") {
      return route.path === path;
    } else if (route.path instanceof RegExp) {
      return route.path.test(path);
    }

    return false;
  });

  if (matchedRoute) {
    const allowedRoles = matchedRoute.roles;
  
    if (!isAuthenticated) {
      return NextResponse.redirect(new URL("/login", request.nextUrl));
    }
  
    if (!userRole || !allowedRoles.includes(userRole)) {
      return NextResponse.redirect(new URL("/not-found", request.nextUrl));
    }
  }
  
  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|.*\\.png$).*)"],
};